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
from src.model_utils import FLEXCAREModel, apply_star_to_logits
from src.utils import set_seed, ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        type=str,
        default="flex_care",
        choices=["baseline", "flex", "flex_care"],
    )

    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--prompt_json", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ViT-B/16")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # ===== FLEX =====
    parser.add_argument("--rank_hidden_dim", type=int, default=256)
    parser.add_argument("--rank_dropout", type=float, default=0.1)
    parser.add_argument(
        "--local_hidden_dim",
        "--evidence_hidden_dim",
        dest="evidence_hidden_dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--local_dropout",
        "--evidence_dropout",
        dest="evidence_dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--local_top_r",
        "--evidence_top_r",
        dest="evidence_top_r",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--local_pool_tau",
        "--evidence_pool_tau",
        dest="evidence_pool_tau",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--glsim_weight",
        "--evidence_dev_weight",
        dest="evidence_dev_weight",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--rank_weight",
        "--evidence_rank_weight",
        dest="evidence_rank_weight",
        type=float,
        default=0.7,
    )
    parser.add_argument("--spatial_smooth_mu", type=float, default=0.15)
    parser.add_argument(
        "--max_local_weight",
        "--max_evidence_weight",
        dest="max_evidence_weight",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--local_init_weight",
        "--init_evidence_weight",
        dest="init_evidence_weight",
        type=float,
        default=0.05,
    )

    # ===== CARE =====
    parser.add_argument(
        "--care_mode",
        type=str,
        default="prob_gap",
        choices=["original", "prob_gap", "logit_temp_scaled", "logit_mean_norm"],
    )
    parser.add_argument(
        "--align_eta",
        "--care_class_penalty_weight",
        dest="care_class_penalty_weight",
        type=float,
        default=0.05,
    )
    parser.add_argument("--care_js_temp", type=float, default=1.0)
    parser.add_argument("--care_reliability_beta_class", type=float, default=1.0)
    parser.add_argument("--care_reliability_beta_js", type=float, default=1.0)
    parser.add_argument("--care_disagreement_temp", type=float, default=10.0)

    # ===== STAR =====
    parser.add_argument(
        "--disable_star",
        action="store_true",
        help="if set, evaluate FLEX+CARE only; otherwise apply STAR",
    )
    parser.add_argument("--star_risk_weight_reliability", type=float, default=1.0)
    parser.add_argument("--star_risk_weight_uncertainty", type=float, default=1.0)
    parser.add_argument("--star_risk_weight_seen_bias", type=float, default=1.0)
    parser.add_argument("--star_dynamic_seen_suppression_kappa", type=float, default=1.0)
    parser.add_argument("--star_triage_tau_accept", type=float, default=0.40)
    parser.add_argument("--star_triage_tau_alert", type=float, default=0.70)

    # ===== prompt =====
    parser.add_argument("--expected_num_prompts", type=int, default=1)

    # ===== checkpoint / eval =====
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs/results/flex_care_star_eval")
    parser.add_argument("--measure_eval_time", action="store_true")
    parser.add_argument("--save_per_sample_outputs", action="store_true")

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
        "macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0),
            2,
        ),
        "macro_auc": round(macro_auc_ovr_safe(y_true, probs, num_classes=num_classes), 2),
    }
    return result


def expected_calibration_error(logits, labels, n_bins=15):
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros(1, device=logits.device)

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        in_bin = (conf > lower) & (conf <= upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = correct[in_bin].float().mean()
            avg_conf = conf[in_bin].mean()
            ece += torch.abs(avg_conf - acc_in_bin) * prop_in_bin

    return float(ece.item())


def multiclass_brier_score(logits, labels):
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
    score = torch.mean(torch.sum((probs - one_hot) ** 2, dim=1))
    return float(score.item())


def nll_score(logits, labels):
    return float(F.cross_entropy(logits, labels).item())


def selective_metrics(logits, labels, actions):
    pred = logits.argmax(dim=1)
    correct = pred.eq(labels)

    accept_mask = actions == 0
    defer_mask = actions == 1
    alert_mask = actions == 2

    coverage = float(accept_mask.float().mean().item())
    defer_ratio = float(defer_mask.float().mean().item())
    alert_ratio = float(alert_mask.float().mean().item())

    if accept_mask.any():
        accept_acc = float(correct[accept_mask].float().mean().item() * 100.0)
    else:
        accept_acc = None

    return {
        "coverage_accept_ratio": round(coverage, 6),
        "defer_ratio": round(defer_ratio, 6),
        "alert_ratio": round(alert_ratio, 6),
        "accept_subset_accuracy": round(accept_acc, 2) if accept_acc is not None else None,
    }


def maybe_load_ckpt(model, ckpt_path, device):
    if not ckpt_path:
        return None

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print(f"Loaded checkpoint from: {ckpt_path}")

    if "epoch" in ckpt:
        print(f"Checkpoint epoch: {ckpt['epoch']}")
    if "best_h" in ckpt:
        print(f"Checkpoint best_h: {ckpt['best_h']}")
    if "args" in ckpt and isinstance(ckpt["args"], dict):
        print(f"Checkpoint care_mode: {ckpt['args'].get('care_mode', 'unknown')}")

    return ckpt


@torch.no_grad()
def collect_outputs_for_loader(
    clip_model,
    model,
    loader,
    text_bank,
    seen_mask,
    apply_star,
    args,
    device,
):
    clip_model.eval()
    model.eval()

    all_labels = []
    all_logits_before_star = []
    all_logits_final = []
    all_reliability = []
    all_risk = []
    all_actions = []

    total_time = 0.0
    total_images = 0

    care_disagreement_top1_sum = 0.0
    care_js_div_sum = 0.0
    care_reliability_sum = 0.0

    star_risk_sum = 0.0
    star_accept_sum = 0.0
    star_defer_sum = 0.0
    star_alert_sum = 0.0

    total_samples = 0

    if args.measure_eval_time and device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        labels = labels.to(device, non_blocking=True)
        batch_size = labels.size(0)

        start_t = time.perf_counter()

        global_features, patch_tokens = encode_images_with_patches(clip_model, images, device)
        logits_before_star, aux = model(
            global_feats=global_features,
            patch_tokens=patch_tokens,
            text_bank=text_bank,
        )

        reliability = aux["reliability"]

        if apply_star:
            final_logits, risk, actions, star_aux = apply_star_to_logits(
                calibrated_logits=aux["calibrated_logits"],
                reliability=reliability,
                seen_mask=seen_mask,
                risk_weight_reliability=args.star_risk_weight_reliability,
                risk_weight_uncertainty=args.star_risk_weight_uncertainty,
                risk_weight_seen_bias=args.star_risk_weight_seen_bias,
                dynamic_seen_suppression_kappa=args.star_dynamic_seen_suppression_kappa,
                triage_tau_accept=args.star_triage_tau_accept,
                triage_tau_alert=args.star_triage_tau_alert,
            )

            star_risk_sum += float(risk.mean().detach().cpu().item()) * batch_size
            star_accept_sum += float((actions == 0).float().mean().detach().cpu().item()) * batch_size
            star_defer_sum += float((actions == 1).float().mean().detach().cpu().item()) * batch_size
            star_alert_sum += float((actions == 2).float().mean().detach().cpu().item()) * batch_size
        else:
            final_logits = logits_before_star
            risk = torch.zeros(batch_size, device=labels.device)
            actions = torch.zeros(batch_size, dtype=torch.long, device=labels.device)

        end_t = time.perf_counter()

        if args.measure_eval_time:
            total_time += end_t - start_t
            total_images += batch_size

        if "care_disagreement_top1_mean" in aux:
            care_disagreement_top1_sum += float(aux["care_disagreement_top1_mean"]) * batch_size
            care_js_div_sum += float(aux["care_js_div_mean"]) * batch_size
            care_reliability_sum += float(aux["care_reliability_mean"]) * batch_size

        total_samples += batch_size

        all_labels.append(labels.detach().cpu())
        all_logits_before_star.append(logits_before_star.detach().cpu())
        all_logits_final.append(final_logits.detach().cpu())
        all_reliability.append(reliability.detach().cpu())
        all_risk.append(risk.detach().cpu())
        all_actions.append(actions.detach().cpu())

    labels = torch.cat(all_labels, dim=0)
    logits_before_star = torch.cat(all_logits_before_star, dim=0)
    logits_final = torch.cat(all_logits_final, dim=0)
    reliability = torch.cat(all_reliability, dim=0)
    risk = torch.cat(all_risk, dim=0)
    actions = torch.cat(all_actions, dim=0)

    metrics = {
        "ece": round(expected_calibration_error(logits_final, labels), 6),
        "nll": round(nll_score(logits_final, labels), 6),
        "brier": round(multiclass_brier_score(logits_final, labels), 6),
        "avg_seconds_per_image": (
            round(float(total_time / max(total_images, 1)), 6)
            if args.measure_eval_time
            else None
        ),
        "care_disagreement_top1_mean": round(care_disagreement_top1_sum / max(total_samples, 1), 6),
        "care_js_div_mean": round(care_js_div_sum / max(total_samples, 1), 6),
        "care_reliability_mean": round(care_reliability_sum / max(total_samples, 1), 6),
    }

    if args.measure_eval_time and device == "cuda" and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        metrics["peak_memory_mb"] = round(float(peak_memory_mb), 2)
    else:
        metrics["peak_memory_mb"] = None

    if apply_star:
        metrics.update(
            {
                "star_risk_mean": round(star_risk_sum / max(total_samples, 1), 6),
                "star_accept_ratio": round(star_accept_sum / max(total_samples, 1), 6),
                "star_defer_ratio": round(star_defer_sum / max(total_samples, 1), 6),
                "star_alert_ratio": round(star_alert_sum / max(total_samples, 1), 6),
            }
        )
        metrics.update(selective_metrics(logits_final, labels, actions))
    else:
        metrics.update(
            {
                "star_risk_mean": None,
                "star_accept_ratio": None,
                "star_defer_ratio": None,
                "star_alert_ratio": None,
                "coverage_accept_ratio": None,
                "defer_ratio": None,
                "alert_ratio": None,
                "accept_subset_accuracy": None,
            }
        )

    per_sample = {
        "labels": labels,
        "logits_before_star": logits_before_star,
        "logits_final": logits_final,
        "pred_before_star": logits_before_star.argmax(dim=1),
        "pred_final": logits_final.argmax(dim=1),
        "reliability": reliability,
        "risk": risk,
        "actions": actions,
    }

    return logits_final, labels, metrics, per_sample


@torch.no_grad()
def run_full_evaluation(
    clip_model,
    model,
    base_test_loader,
    new_test_loader,
    text_bank,
    seen_mask,
    apply_star,
    args,
    device,
):
    base_logits, base_labels, base_metrics, base_per_sample = collect_outputs_for_loader(
        clip_model=clip_model,
        model=model,
        loader=base_test_loader,
        text_bank=text_bank,
        seen_mask=seen_mask,
        apply_star=apply_star,
        args=args,
        device=device,
    )

    new_logits, new_labels, new_metrics, new_per_sample = collect_outputs_for_loader(
        clip_model=clip_model,
        model=model,
        loader=new_test_loader,
        text_bank=text_bank,
        seen_mask=seen_mask,
        apply_star=apply_star,
        args=args,
        device=device,
    )

    seen_result = evaluate_split("seen(base_test)", base_logits, base_labels)
    unseen_result = evaluate_split("unseen(new_test)", new_logits, new_labels)
    h_score = round(harmonic_mean(seen_result["top1"], unseen_result["top1"]), 2)

    per_sample = {
        "base_test": base_per_sample,
        "new_test": new_per_sample,
    }

    return seen_result, unseen_result, h_score, base_metrics, new_metrics, per_sample


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    apply_star = (not args.disable_star) and (args.variant == "flex_care")

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    print(f"Using device: {device}")
    print(f"CARE mode: {args.care_mode}")
    print(f"Apply STAR: {apply_star}")

    print("Building class splits...")
    seen_classes, unseen_classes, all_classes = build_class_splits(args.data_root)

    print(f"Seen classes   : {len(seen_classes)}")
    print(f"Unseen classes : {len(unseen_classes)}")
    print(f"All classes    : {len(all_classes)}")

    class_to_global_idx = {name: i for i, name in enumerate(all_classes)}
    seen_global_indices = torch.tensor(
        [class_to_global_idx[name] for name in seen_classes],
        dtype=torch.long,
        device=device,
    )
    seen_mask = torch.zeros(len(all_classes), dtype=torch.bool, device=device)
    seen_mask[seen_global_indices] = True

    print("Loading CLIP...")
    clip_model, preprocess = clip.load(args.model_name, device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    if "ViT" not in args.model_name:
        raise ValueError("This script requires a ViT backbone, e.g. ViT-B/16 or ViT-B/32.")

    print("Building datasets...")
    base_test_dataset = PlantWildDataset(
        root=os.path.join(args.data_root, "base_test"),
        all_classes=all_classes,
        transform=preprocess,
    )
    new_test_dataset = PlantWildDataset(
        root=os.path.join(args.data_root, "new_test"),
        all_classes=all_classes,
        transform=preprocess,
    )

    base_test_loader = DataLoader(
        base_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    new_test_loader = DataLoader(
        new_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    print("Loading prompt json...")
    prompt_dict = load_prompt_dict(args.prompt_json)

    print("Encoding single-prompt text bank...")
    all_text_bank = encode_prompts_per_class_fixed_count(
        model=clip_model,
        prompt_dict=prompt_dict,
        classnames=all_classes,
        device=device,
        expected_num_prompts=args.expected_num_prompts,
    ).float().to(device)

    if all_text_bank.shape[1] != 1:
        raise ValueError(
            f"This script expects single prompt per class. "
            f"Got expected_num_prompts={args.expected_num_prompts}, actual P={all_text_bank.shape[1]}"
        )

    all_text_bank = all_text_bank[:, 0, :]
    all_text_bank = F.normalize(all_text_bank, dim=-1)

    feat_dim = all_text_bank.shape[-1]
    print(f"Feature dim: {feat_dim}")
    print(f"CARE class penalty: {args.care_class_penalty_weight}")

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
        care_mode=args.care_mode,
        care_class_penalty_weight=args.care_class_penalty_weight,
        care_js_temp=args.care_js_temp,
        care_reliability_beta_class=args.care_reliability_beta_class,
        care_reliability_beta_js=args.care_reliability_beta_js,
        care_disagreement_temp=args.care_disagreement_temp,
    ).to(device)

    if args.variant != "baseline":
        if not args.resume_ckpt:
            raise ValueError("resume_ckpt is required for non-baseline evaluation.")
        maybe_load_ckpt(model, args.resume_ckpt, device)

    seen_result, unseen_result, h_score, base_metrics, new_metrics, per_sample = run_full_evaluation(
        clip_model=clip_model,
        model=model,
        base_test_loader=base_test_loader,
        new_test_loader=new_test_loader,
        text_bank=all_text_bank,
        seen_mask=seen_mask,
        apply_star=apply_star,
        args=args,
        device=device,
    )

    final_result = {
        "method": "flex_care_star_eval",
        "variant": args.variant,
        "apply_star": apply_star,
        "model_name": args.model_name,
        "prompt_json": args.prompt_json,
        "expected_num_prompts": args.expected_num_prompts,
        "resume_ckpt": args.resume_ckpt,
        "care_mode": args.care_mode,
        "care_disagreement_temp": args.care_disagreement_temp,
        "care_class_penalty_weight": args.care_class_penalty_weight,
        "care_js_temp": args.care_js_temp,
        "care_reliability_beta_class": args.care_reliability_beta_class,
        "care_reliability_beta_js": args.care_reliability_beta_js,
        "star_risk_weight_reliability": args.star_risk_weight_reliability,
        "star_risk_weight_uncertainty": args.star_risk_weight_uncertainty,
        "star_risk_weight_seen_bias": args.star_risk_weight_seen_bias,
        "star_dynamic_seen_suppression_kappa": args.star_dynamic_seen_suppression_kappa,
        "star_triage_tau_accept": args.star_triage_tau_accept,
        "star_triage_tau_alert": args.star_triage_tau_alert,
        "num_seen_classes": len(seen_classes),
        "num_unseen_classes": len(unseen_classes),
        "num_all_classes": len(all_classes),
        "seen": seen_result,
        "unseen": unseen_result,
        "h_score": h_score,
        "base_eval_metrics": base_metrics,
        "new_eval_metrics": new_metrics,
        "args": vars(args),
    }

    final_json = os.path.join(args.output_dir, "eval_result.json")
    save_json(final_result, final_json)

    if args.save_per_sample_outputs:
        per_sample_path = os.path.join(args.output_dir, "per_sample_outputs.pt")
        torch.save(per_sample, per_sample_path)
        print(f"Saved per-sample outputs to: {per_sample_path}")

    print("\n===== Evaluation Result =====")
    print(json.dumps(final_result, indent=2, ensure_ascii=False))
    print(f"Saved to: {final_json}")


if __name__ == "__main__":
    main()