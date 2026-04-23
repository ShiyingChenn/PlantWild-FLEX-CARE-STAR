import os
import sys
import json
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

CLIP_ROOT = None
for cand in ["clip-main", "CLIP-main"]:
    p = os.path.join(PROJECT_ROOT, cand)
    if os.path.isdir(p):
        CLIP_ROOT = p
        break

if CLIP_ROOT is None:
    raise FileNotFoundError(
        "Cannot find clip-main/ or CLIP-main/ under project root. "
        "Please check your folder name."
    )

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if CLIP_ROOT not in sys.path:
    sys.path.insert(0, CLIP_ROOT)

import clip

from src.dataset import PlantWildDataset, build_class_splits
from src.prompt_utils import load_prompt_dict, encode_prompts_per_class_fixed_count
from src.model_utils import FLEXCAREModel
from src.utils import set_seed, ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--variant", type=str, default="flex_care",
                        choices=["baseline", "flex", "flex_care"])

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--prompt_json", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ViT-B/16")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--rank_hidden_dim", type=int, default=256)
    parser.add_argument("--rank_dropout", type=float, default=0.1)
    parser.add_argument("--local_hidden_dim", "--evidence_hidden_dim", dest="evidence_hidden_dim", type=int, default=512)
    parser.add_argument("--local_dropout", "--evidence_dropout", dest="evidence_dropout", type=float, default=0.1)
    parser.add_argument("--local_top_r", "--evidence_top_r", dest="evidence_top_r", type=int, default=16)
    parser.add_argument("--local_pool_tau", "--evidence_pool_tau", dest="evidence_pool_tau", type=float, default=0.5)
    parser.add_argument("--glsim_weight", "--evidence_dev_weight", dest="evidence_dev_weight", type=float, default=0.3)
    parser.add_argument("--rank_weight", "--evidence_rank_weight", dest="evidence_rank_weight", type=float, default=0.7)
    parser.add_argument("--spatial_smooth_mu", type=float, default=0.15)
    parser.add_argument("--max_local_weight", "--max_evidence_weight", dest="max_evidence_weight", type=float, default=0.10)
    parser.add_argument("--local_init_weight", "--init_evidence_weight", dest="init_evidence_weight", type=float, default=0.05)

    parser.add_argument("--align_eta", "--care_class_penalty_weight", dest="care_class_penalty_weight", type=float, default=0.05)
    parser.add_argument("--care_js_temp", type=float, default=1.0)
    parser.add_argument("--care_reliability_beta_class", type=float, default=1.0)
    parser.add_argument("--care_reliability_beta_js", type=float, default=1.0)
    parser.add_argument("--train_with_calibrated_logits", action="store_true",
                        help="if set, train with CARE-calibrated logits; otherwise train with fused logits")

    parser.add_argument("--expected_num_prompts", type=int, default=1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./outputs/results/flex_care")

    parser.add_argument("--measure_eval_time", action="store_true")

    return parser.parse_args()


def get_amp_context(device):
    if device == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def encode_images_with_patches(model, images, device):
    images = images.to(device, non_blocking=True)

    with get_amp_context(device):
        outputs = model.encode_image(images, return_patch_tokens=True)

    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise RuntimeError(
            "model.encode_image(..., return_patch_tokens=True) must return "
            "(global_feat, patch_tokens). Please check clip/model.py."
        )

    global_features, patch_tokens = outputs

    if patch_tokens is None:
        raise RuntimeError("patch_tokens is None. This script requires a ViT backbone.")

    global_features = F.normalize(global_features.float(), dim=-1)
    patch_tokens = F.normalize(patch_tokens.float(), dim=-1)
    return global_features, patch_tokens


def topk_accuracy(logits, labels, k=1):
    k = min(k, logits.size(1))
    topk_idx = logits.topk(k=k, dim=1).indices
    correct = topk_idx.eq(labels.unsqueeze(1)).any(dim=1).float().mean().item()
    return correct * 100.0


def macro_auc_ovr_safe(y_true, y_score, num_classes):
    auc_list = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(np.int32)
        if y_bin.max() == 0:
            continue
        if y_bin.min() == 1:
            continue
        try:
            auc = roc_auc_score(y_bin, y_score[:, c])
            auc_list.append(auc)
        except Exception:
            continue

    if len(auc_list) == 0:
        return 0.0
    return float(np.mean(auc_list) * 100.0)


def harmonic_mean(seen_acc, unseen_acc, eps=1e-12):
    return float(2.0 * seen_acc * unseen_acc / (seen_acc + unseen_acc + eps))


def evaluate_split(split_name, logits, labels):
    probs = F.softmax(logits, dim=1).cpu().numpy()
    y_true = labels.cpu().numpy()
    y_pred = logits.argmax(dim=1).cpu().numpy()

    num_classes = logits.size(1)

    result = {
        "split": split_name,
        "top1": round(topk_accuracy(logits, labels, k=1), 2),
        "top3": round(topk_accuracy(logits, labels, k=3), 2),
        "top5": round(topk_accuracy(logits, labels, k=5), 2),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0), 2),
        "macro_auc": round(macro_auc_ovr_safe(y_true, probs, num_classes=num_classes), 2),
    }
    return result


def extract_scalar_aux(aux):
    scalar_aux = {}
    for key, value in aux.items():
        if isinstance(value, (int, float)):
            scalar_aux[key] = float(value)
        elif torch.is_tensor(value) and value.numel() == 1:
            scalar_aux[key] = float(value.detach().cpu().item())
    return scalar_aux


@torch.no_grad()
def collect_logits_for_loader(clip_model, model, loader, text_bank, device, measure_eval_time=False):
    clip_model.eval()
    model.eval()

    all_logits = []
    all_labels = []

    total_time = 0.0
    total_images = 0
    scalar_stat_sum = {}
    total_samples = 0

    if measure_eval_time and device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        start_t = time.perf_counter()
        global_features, patch_tokens = encode_images_with_patches(clip_model, images, device)
        logits, aux = model(global_features, patch_tokens, text_bank)
        end_t = time.perf_counter()

        if measure_eval_time:
            total_time += (end_t - start_t)
            total_images += batch_size

        scalar_aux = extract_scalar_aux(aux)
        for key, value in scalar_aux.items():
            scalar_stat_sum[key] = scalar_stat_sum.get(key, 0.0) + value * batch_size

        total_samples += batch_size
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    stats = {}
    if total_samples > 0:
        stats = {k: round(v / total_samples, 6) for k, v in scalar_stat_sum.items()}

    stats["avg_seconds_per_image"] = round(float(total_time / max(total_images, 1)), 6) if measure_eval_time else None
    if measure_eval_time and device == "cuda" and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        stats["peak_memory_mb"] = round(float(peak_memory_mb), 2)
    else:
        stats["peak_memory_mb"] = None

    return all_logits, all_labels, stats


@torch.no_grad()
def run_full_evaluation(
    clip_model,
    model,
    base_test_loader,
    new_test_loader,
    text_bank,
    device,
    measure_eval_time=False
):
    base_logits, base_labels, base_stats = collect_logits_for_loader(
        clip_model=clip_model,
        model=model,
        loader=base_test_loader,
        text_bank=text_bank,
        device=device,
        measure_eval_time=measure_eval_time
    )

    new_logits, new_labels, new_stats = collect_logits_for_loader(
        clip_model=clip_model,
        model=model,
        loader=new_test_loader,
        text_bank=text_bank,
        device=device,
        measure_eval_time=measure_eval_time
    )

    seen_result = evaluate_split("seen(base_test)", base_logits, base_labels)
    unseen_result = evaluate_split("unseen(new_test)", new_logits, new_labels)
    h_score = round(harmonic_mean(seen_result["top1"], unseen_result["top1"]), 2)

    eval_stats = {
        "base_eval": base_stats,
        "new_eval": new_stats
    }
    return seen_result, unseen_result, h_score, eval_stats


def format_epoch_result(epoch, epochs, train_loss, seen_result, unseen_result, h_score, evidence_weight):
    msg = (
        f"[Epoch {epoch:03d}/{epochs:03d}] "
        f"TrainLoss={train_loss:.4f} | "
        f"EvidenceW={evidence_weight:.4f} | "
        f"Seen: Top1={seen_result['top1']:.2f} Top3={seen_result['top3']:.2f} "
        f"Top5={seen_result['top5']:.2f} F1={seen_result['macro_f1']:.2f} AUC={seen_result['macro_auc']:.2f} | "
        f"Unseen: Top1={unseen_result['top1']:.2f} Top3={unseen_result['top3']:.2f} "
        f"Top5={unseen_result['top5']:.2f} F1={unseen_result['macro_f1']:.2f} AUC={unseen_result['macro_auc']:.2f} | "
        f"H={h_score:.2f}"
    )
    return msg


def run_eval_only(
    args,
    clip_model,
    model,
    base_test_loader,
    new_test_loader,
    all_text_bank,
    seen_classes,
    unseen_classes,
    all_classes,
    output_dir,
    device
):
    seen_result, unseen_result, h_score, eval_stats = run_full_evaluation(
        clip_model=clip_model,
        model=model,
        base_test_loader=base_test_loader,
        new_test_loader=new_test_loader,
        text_bank=all_text_bank,
        device=device,
        measure_eval_time=args.measure_eval_time
    )

    result = {
        "method": "baseline",
        "variant": args.variant,
        "best_epoch": 0,
        "best_result": {
            "epoch": 0,
            "train_loss": None,
            "evidence_weight": 0.0,
            "seen": seen_result,
            "unseen": unseen_result,
            "h_score": h_score,
            "eval_stats": eval_stats
        },
        "num_seen_classes": len(seen_classes),
        "num_unseen_classes": len(unseen_classes),
        "num_all_classes": len(all_classes)
    }

    save_json(result, os.path.join(output_dir, "final_result.json"))
    save_json(result["best_result"], os.path.join(output_dir, "best_h_metrics.json"))

    print("\n===== Baseline Evaluation Finished =====")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")

    print("Building class splits...")
    seen_classes, unseen_classes, all_classes = build_class_splits(args.data_root)

    print(f"Seen classes   : {len(seen_classes)}")
    print(f"Unseen classes : {len(unseen_classes)}")
    print(f"All classes    : {len(all_classes)}")

    class_to_global_idx = {name: i for i, name in enumerate(all_classes)}
    seen_global_indices = torch.tensor(
        [class_to_global_idx[name] for name in seen_classes],
        dtype=torch.long,
        device=device
    )

    global_to_seen = torch.full(
        (len(all_classes),),
        fill_value=-1,
        dtype=torch.long,
        device=device
    )
    for local_idx, global_idx in enumerate(seen_global_indices.tolist()):
        global_to_seen[global_idx] = local_idx

    print("Loading CLIP...")
    clip_model, preprocess = clip.load(args.model_name, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    if "ViT" not in args.model_name:
        raise ValueError("This script requires a ViT backbone, e.g. ViT-B/16 or ViT-B/32.")

    print("Building datasets...")
    base_train_dataset = PlantWildDataset(
        root=os.path.join(args.data_root, "base_train"),
        all_classes=all_classes,
        transform=preprocess
    )
    base_test_dataset = PlantWildDataset(
        root=os.path.join(args.data_root, "base_test"),
        all_classes=all_classes,
        transform=preprocess
    )
    new_test_dataset = PlantWildDataset(
        root=os.path.join(args.data_root, "new_test"),
        all_classes=all_classes,
        transform=preprocess
    )

    train_loader = DataLoader(
        base_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False
    )
    base_test_loader = DataLoader(
        base_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )
    new_test_loader = DataLoader(
        new_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda")
    )

    print("Loading prompt json...")
    prompt_dict = load_prompt_dict(args.prompt_json)

    print("Encoding single-prompt text bank...")
    all_text_bank = encode_prompts_per_class_fixed_count(
        model=clip_model,
        prompt_dict=prompt_dict,
        classnames=all_classes,
        device=device,
        expected_num_prompts=args.expected_num_prompts
    ).float().to(device)

    if all_text_bank.shape[1] != 1:
        raise ValueError(
            f"This script expects single prompt per class. "
            f"Got expected_num_prompts={args.expected_num_prompts}, actual P={all_text_bank.shape[1]}"
        )

    all_text_bank = all_text_bank[:, 0, :]
    all_text_bank = F.normalize(all_text_bank, dim=-1)
    seen_text_bank = all_text_bank.index_select(dim=0, index=seen_global_indices)

    feat_dim = all_text_bank.shape[-1]
    print(f"Feature dim: {feat_dim}")
    print(f"CARE align eta(class penalty): {args.care_class_penalty_weight}")

    model = FLEXCAREModel(
        feat_dim=feat_dim,
        variant=args.variant,
        rank_hidden_dim=args.rank_hidden_dim,
        rank_dropout=args.rank_dropout,
        evidence_hidden_dim=args.evidence_hidden_dim,
        evidence_dropout=args.evidence_dropout,
        evidence_top_r=args.evidence_top_r,
        evidence_pool_tau=args.evidence_pool_tau,
        evidence_dev_weight=args.evidence_dev_weight,
        evidence_rank_weight=args.evidence_rank_weight,
        spatial_smooth_mu=args.spatial_smooth_mu,
        max_evidence_weight=args.max_evidence_weight,
        init_evidence_weight=args.init_evidence_weight,
        care_class_penalty_weight=args.care_class_penalty_weight,
        care_js_temp=args.care_js_temp,
        care_reliability_beta_class=args.care_reliability_beta_class,
        care_reliability_beta_js=args.care_reliability_beta_js
    ).to(device)

    if args.variant == "baseline":
        run_eval_only(
            args=args,
            clip_model=clip_model,
            model=model,
            base_test_loader=base_test_loader,
            new_test_loader=new_test_loader,
            all_text_bank=all_text_bank,
            seen_classes=seen_classes,
            unseen_classes=unseen_classes,
            all_classes=all_classes,
            output_dir=args.output_dir,
            device=device
        )
        return

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    history = []
    best_h = -1.0
    best_epoch = -1
    best_result = None

    print("\n===== Start Training =====")
    for epoch in range(1, args.epochs + 1):
        model.train()
        clip_model.eval()

        running_loss = 0.0
        num_samples = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, labels_global in pbar:
            labels_global = labels_global.to(device, non_blocking=True)
            labels_local = global_to_seen[labels_global]

            if (labels_local < 0).any():
                bad_idx = torch.where(labels_local < 0)[0][:10].tolist()
                raise RuntimeError(
                    f"Found labels not in seen classes at training time. Example batch indices: {bad_idx}"
                )

            with torch.no_grad():
                global_features, patch_tokens = encode_images_with_patches(clip_model, images, device)

            logits, aux = model(
                global_feats=global_features,
                patch_tokens=patch_tokens,
                text_bank=seen_text_bank
            )

            if args.variant == "flex_care" and not args.train_with_calibrated_logits:
                train_logits = aux["fused_logits"]
            else:
                train_logits = logits

            loss = F.cross_entropy(train_logits, labels_local)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = labels_global.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            pbar.set_postfix({
                "loss": f"{(running_loss / max(num_samples, 1)):.4f}",
                "ew": f"{model.get_evidence_weight():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        scheduler.step()

        train_loss = running_loss / max(num_samples, 1)
        evidence_weight = model.get_evidence_weight()

        if epoch % args.eval_every == 0:
            seen_result, unseen_result, h_score, eval_stats = run_full_evaluation(
                clip_model=clip_model,
                model=model,
                base_test_loader=base_test_loader,
                new_test_loader=new_test_loader,
                text_bank=all_text_bank,
                device=device,
                measure_eval_time=args.measure_eval_time
            )

            epoch_record = {
                "epoch": epoch,
                "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
                "train_loss": round(float(train_loss), 6),
                "evidence_weight": round(float(evidence_weight), 6),
                "seen": seen_result,
                "unseen": unseen_result,
                "h_score": h_score,
                "eval_stats": eval_stats
            }
            history.append(epoch_record)

            print(format_epoch_result(
                epoch=epoch,
                epochs=args.epochs,
                train_loss=train_loss,
                seen_result=seen_result,
                unseen_result=unseen_result,
                h_score=h_score,
                evidence_weight=evidence_weight
            ))

            save_json(history, os.path.join(args.output_dir, "history.json"))

            if h_score > best_h:
                best_h = h_score
                best_epoch = epoch
                best_result = epoch_record

                ckpt_path = os.path.join(args.output_dir, f"best_h_{args.variant}.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_h": best_h,
                    "args": vars(args)
                }, ckpt_path)

                save_json(best_result, os.path.join(args.output_dir, "best_h_metrics.json"))

    print("\n===== Training Finished =====")
    if best_result is None:
        print("No evaluation result found. Please check eval_every / epochs.")
        return

    final_result = {
        "method": "flex_care",
        "variant": args.variant,
        "model_name": args.model_name,
        "prompt_json": args.prompt_json,
        "expected_num_prompts": args.expected_num_prompts,
        "train_with_calibrated_logits": args.train_with_calibrated_logits,

        "rank_hidden_dim": args.rank_hidden_dim,
        "rank_dropout": args.rank_dropout,
        "evidence_hidden_dim": args.evidence_hidden_dim,
        "evidence_dropout": args.evidence_dropout,
        "evidence_top_r": args.evidence_top_r,
        "evidence_pool_tau": args.evidence_pool_tau,
        "evidence_dev_weight": args.evidence_dev_weight,
        "evidence_rank_weight": args.evidence_rank_weight,
        "spatial_smooth_mu": args.spatial_smooth_mu,
        "max_evidence_weight": args.max_evidence_weight,
        "init_evidence_weight": args.init_evidence_weight,

        "care_class_penalty_weight": args.care_class_penalty_weight,
        "care_js_temp": args.care_js_temp,
        "care_reliability_beta_class": args.care_reliability_beta_class,
        "care_reliability_beta_js": args.care_reliability_beta_js,

        "num_seen_classes": len(seen_classes),
        "num_unseen_classes": len(unseen_classes),
        "num_all_classes": len(all_classes),
        "best_epoch": best_epoch,
        "best_result": best_result,
        "args": vars(args)
    }

    final_json = os.path.join(args.output_dir, "final_result.json")
    save_json(final_result, final_json)

    print("\n===== Best H-Score Result =====")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))
    print(f"Saved to: {final_json}")


if __name__ == "__main__":
    main()