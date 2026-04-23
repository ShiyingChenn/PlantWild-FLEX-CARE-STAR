import json
import torch
import clip
import torch.nn as nn
import torch.nn.functional as F

def build_single_prompts(classnames, template):
    return [template.format(name) for name in classnames]


def load_prompt_dict(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        prompt_dict = json.load(f)
    return prompt_dict


# 兼容旧名字
load_prompt_json = load_prompt_dict


@torch.no_grad()
def encode_text_prompts(model, prompts, device, batch_size=256):
    """
    prompts: list[str] or str
    return: [N, D]
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    all_features = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokens = clip.tokenize(batch_prompts).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        all_features.append(text_features.float())

    return torch.cat(all_features, dim=0)


@torch.no_grad()
def encode_prompts_per_class(
    model,
    prompt_dict,
    classnames,
    device,
    expected_num_prompts=None,
    batch_size=256
):
    """
    返回:
        text_features_all: [C, P, D]
    """
    all_class_prompt_features = []

    for cls_name in classnames:
        if cls_name not in prompt_dict:
            raise KeyError(f"Class '{cls_name}' not found in prompt json.")

        prompts = prompt_dict[cls_name]
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError(f"Prompts for class '{cls_name}' must be a non-empty list.")

        if expected_num_prompts is not None and len(prompts) != expected_num_prompts:
            raise ValueError(
                f"Class '{cls_name}' has {len(prompts)} prompts, expected {expected_num_prompts}."
            )

        class_text_features = encode_text_prompts(
            model=model,
            prompts=prompts,
            device=device,
            batch_size=batch_size
        )  # [P, D]

        all_class_prompt_features.append(class_text_features)

    # 这里默认每类 prompt 数相同（你现在是 50）
    text_features_all = torch.stack(all_class_prompt_features, dim=0)  # [C, P, D]
    return text_features_all


@torch.no_grad()
def encode_prompts_per_class_fixed_count(
    model,
    prompt_dict,
    classnames,
    device,
    expected_num_prompts=50,
    batch_size=256
):
    return encode_prompts_per_class(
        model=model,
        prompt_dict=prompt_dict,
        classnames=classnames,
        device=device,
        expected_num_prompts=expected_num_prompts,
        batch_size=batch_size
    )


def average_prompt_fusion(text_features_all):
    """
    text_features_all: [C, P, D]
    return: [C, D]
    """
    fused = text_features_all.mean(dim=1)
    fused = fused / fused.norm(dim=-1, keepdim=True)
    return fused


def average_prompt_fusion_fixed(text_features_all):
    return average_prompt_fusion(text_features_all)


def build_anchor_prompts_from_first(prompt_dict, classnames, prompt_index=0):
    """
    直接把每类的第 prompt_index 条 prompt 作为 anchor prompt
    """
    anchor_prompts = []
    for cls_name in classnames:
        if cls_name not in prompt_dict:
            raise KeyError(f"Class '{cls_name}' not found in prompt json.")

        prompts = prompt_dict[cls_name]
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError(f"Prompts for class '{cls_name}' must be a non-empty list.")

        if prompt_index >= len(prompts):
            raise IndexError(
                f"Class '{cls_name}' only has {len(prompts)} prompts, "
                f"but prompt_index={prompt_index}."
            )

        anchor_prompts.append(prompts[prompt_index])

    return anchor_prompts


def build_anchor_prompts_from_template(classnames, template):
    return [template.format(name) for name in classnames]


def select_topk_prompts_by_centrality(text_features_all, k):
    """
    text_features_all: [C, P, D]
    return:
        selected_features: [C, k, D]
        selected_indices: [C, k]
        selected_scores: [C, k]
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > text_features_all.shape[1]:
        raise ValueError(f"k={k} is larger than num_prompts={text_features_all.shape[1]}")

    centroids = text_features_all.mean(dim=1)  # [C, D]
    centroids = centroids / centroids.norm(dim=-1, keepdim=True)

    scores = torch.sum(text_features_all * centroids.unsqueeze(1), dim=-1)  # [C, P]

    selected_scores, selected_indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)

    gather_idx = selected_indices.unsqueeze(-1).expand(-1, -1, text_features_all.shape[-1])
    selected_features = torch.gather(text_features_all, dim=1, index=gather_idx)

    return selected_features, selected_indices, selected_scores


class LearnablePromptFusion(nn.Module):
    """
    兼容你原来的旧实验
    """

    def __init__(self, num_classes, num_prompts):
        super().__init__()
        self.prompt_logits = nn.Parameter(torch.zeros(num_classes, num_prompts))

    def forward(self, text_features_all):
        """
        text_features_all: [C, P, D]
        return:
            fused: [C, D]
            weights: [C, P]
        """
        weights = torch.softmax(self.prompt_logits, dim=-1)
        fused = (weights.unsqueeze(-1) * text_features_all).sum(dim=1)
        fused = fused / fused.norm(dim=-1, keepdim=True)
        return fused, weights

@torch.no_grad()
def select_aux_prompts_by_anchor_diversity(
    text_features_all,
    anchor_index=0,
    k=2,
    sim_low=0.55,
    sim_high=0.90
):
    """
    从每类 prompt bank 中，选择与 anchor “中等相似”的辅助 prompt。
    目的：
      - 不选最像的（避免同义重复）
      - 不选最不像的（避免偏题）
      - 选中间相似度的，作为互补 prompt

    Args:
        text_features_all: [C, P, D]
        anchor_index: int
        k: int
        sim_low: float
        sim_high: float

    Returns:
        aux_features: [C, k, D]
        aux_indices:  [C, k]
        anchor_bank:  [C, D]
        sims:         [C, P]
    """
    if text_features_all.ndim != 3:
        raise ValueError("text_features_all must be [C, P, D]")

    C, P, D = text_features_all.shape
    if not (0 <= anchor_index < P):
        raise IndexError(f"anchor_index={anchor_index} out of range for P={P}")
    if k <= 0 or k >= P:
        raise ValueError(f"k must be in [1, P-1], got k={k}, P={P}")

    text_features_all = F.normalize(text_features_all, dim=-1)
    anchor_bank = text_features_all[:, anchor_index, :]  # [C, D]

    # 与 anchor 的余弦相似度
    sims = torch.sum(text_features_all * anchor_bank.unsqueeze(1), dim=-1)  # [C, P]

    all_aux_indices = []
    all_aux_features = []

    for c in range(C):
        sim_c = sims[c].clone()  # [P]
        sim_c[anchor_index] = -1.0  # 排除 anchor 自己

        # 优先取“中等相似度”的 prompt
        candidate_mask = (sim_c >= sim_low) & (sim_c <= sim_high)
        candidate_indices = torch.where(candidate_mask)[0]

        if candidate_indices.numel() >= k:
            # 按“离中间点最近”排序
            center = (sim_low + sim_high) / 2.0
            candidate_sims = sim_c[candidate_indices]
            order = torch.argsort(torch.abs(candidate_sims - center), descending=False)
            chosen_idx = candidate_indices[order[:k]]
        else:
            # fallback: 选最接近 0.75 的 k 条
            target = 0.75
            valid_indices = torch.arange(P, device=text_features_all.device)
            valid_indices = valid_indices[valid_indices != anchor_index]
            valid_sims = sim_c[valid_indices]
            order = torch.argsort(torch.abs(valid_sims - target), descending=False)
            chosen_idx = valid_indices[order[:k]]

        chosen_feat = text_features_all[c, chosen_idx, :]  # [k, D]
        all_aux_indices.append(chosen_idx)
        all_aux_features.append(chosen_feat)

    aux_indices = torch.stack(all_aux_indices, dim=0)      # [C, k]
    aux_features = torch.stack(all_aux_features, dim=0)    # [C, k, D]

    return aux_features, aux_indices, anchor_bank, sims


@torch.no_grad()
def refine_text_prototypes(anchor_bank, aux_bank, gamma=0.1):
    """
    用少量辅助 prompt 对 anchor 做轻量精炼。

    Args:
        anchor_bank: [C, D]
        aux_bank:    [C, K, D]
        gamma: float

    Returns:
        refined_bank: [C, D]
    """
    if anchor_bank.ndim != 2:
        raise ValueError("anchor_bank must be [C, D]")
    if aux_bank.ndim != 3:
        raise ValueError("aux_bank must be [C, K, D]")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    anchor_bank = F.normalize(anchor_bank, dim=-1)
    aux_bank = F.normalize(aux_bank, dim=-1)

    aux_mean = aux_bank.mean(dim=1)  # [C, D]
    aux_mean = F.normalize(aux_mean, dim=-1)

    refined_bank = (1.0 - gamma) * anchor_bank + gamma * aux_mean
    refined_bank = F.normalize(refined_bank, dim=-1)

    return refined_bank


@torch.no_grad()
def build_refined_text_bank_from_prompt_dict(
    model,
    prompt_dict,
    classnames,
    device,
    expected_num_prompts=50,
    anchor_index=0,
    aux_k=2,
    gamma=0.1,
    sim_low=0.55,
    sim_high=0.90,
    batch_size=256
):
    """
    一步式构建 refined text bank

    Returns:
        refined_bank: [C, D]
        aux_indices:  [C, K]
        anchor_bank:  [C, D]
        text_features_all: [C, P, D]
        sims: [C, P]
    """
    text_features_all = encode_prompts_per_class_fixed_count(
        model=model,
        prompt_dict=prompt_dict,
        classnames=classnames,
        device=device,
        expected_num_prompts=expected_num_prompts,
        batch_size=batch_size
    )  # [C, P, D]

    aux_bank, aux_indices, anchor_bank, sims = select_aux_prompts_by_anchor_diversity(
        text_features_all=text_features_all,
        anchor_index=anchor_index,
        k=aux_k,
        sim_low=sim_low,
        sim_high=sim_high
    )

    refined_bank = refine_text_prototypes(
        anchor_bank=anchor_bank,
        aux_bank=aux_bank,
        gamma=gamma
    )  # [C, D]

    return refined_bank, aux_indices, anchor_bank, text_features_all, sims

@torch.no_grad()
def select_aux_prompt_indices_by_anchor_diversity(
    text_features_all,
    anchor_index=0,
    k=2,
    sim_low=0.55,
    sim_high=0.90
):
    """
    根据与 anchor 的中等相似度，选择少量辅助 prompt。
    不选最像的（太重复），也不选最不像的（太偏题）。

    Args:
        text_features_all: [C, P, D]
        anchor_index: int
        k: int
        sim_low: float
        sim_high: float

    Returns:
        aux_indices: [C, k]
        sims: [C, P]
    """
    if text_features_all.ndim != 3:
        raise ValueError("text_features_all must be [C, P, D]")

    C, P, D = text_features_all.shape
    if not (0 <= anchor_index < P):
        raise IndexError(f"anchor_index={anchor_index} out of range for P={P}")
    if k <= 0 or k >= P:
        raise ValueError(f"k must be in [1, P-1], got k={k}, P={P}")

    text_features_all = F.normalize(text_features_all, dim=-1)
    anchor_bank = text_features_all[:, anchor_index, :]  # [C, D]
    sims = torch.sum(text_features_all * anchor_bank.unsqueeze(1), dim=-1)  # [C, P]

    all_aux_indices = []

    for c in range(C):
        sim_c = sims[c].clone()
        sim_c[anchor_index] = -1.0

        candidate_mask = (sim_c >= sim_low) & (sim_c <= sim_high)
        candidate_indices = torch.where(candidate_mask)[0]

        if candidate_indices.numel() >= k:
            center = (sim_low + sim_high) / 2.0
            candidate_sims = sim_c[candidate_indices]
            order = torch.argsort(torch.abs(candidate_sims - center), descending=False)
            chosen_idx = candidate_indices[order[:k]]
        else:
            target = 0.75
            valid_indices = torch.arange(P, device=text_features_all.device)
            valid_indices = valid_indices[valid_indices != anchor_index]
            valid_sims = sim_c[valid_indices]
            order = torch.argsort(torch.abs(valid_sims - target), descending=False)
            chosen_idx = valid_indices[order[:k]]

        all_aux_indices.append(chosen_idx)

    aux_indices = torch.stack(all_aux_indices, dim=0)  # [C, k]
    return aux_indices, sims


@torch.no_grad()
def build_single_and_aux_text_banks(
    model,
    prompt_dict,
    classnames,
    device,
    expected_num_prompts=50,
    anchor_index=0,
    aux_k=2,
    sim_low=0.55,
    sim_high=0.90,
    batch_size=256
):
    """
    构建：
      - single prompt 主 bank
      - aux prompt 辅助 bank

    Returns:
        main_bank: [C, D]
        aux_bank: [C, K, D]
        aux_indices: [C, K]
        text_features_all: [C, P, D]
        sims: [C, P]
    """
    text_features_all = encode_prompts_per_class_fixed_count(
        model=model,
        prompt_dict=prompt_dict,
        classnames=classnames,
        device=device,
        expected_num_prompts=expected_num_prompts,
        batch_size=batch_size
    )  # [C, P, D]

    text_features_all = F.normalize(text_features_all, dim=-1)
    main_bank = text_features_all[:, anchor_index, :]  # [C, D]

    aux_indices, sims = select_aux_prompt_indices_by_anchor_diversity(
        text_features_all=text_features_all,
        anchor_index=anchor_index,
        k=aux_k,
        sim_low=sim_low,
        sim_high=sim_high
    )

    gather_idx = aux_indices.unsqueeze(-1).expand(-1, -1, text_features_all.shape[-1])
    aux_bank = torch.gather(text_features_all, dim=1, index=gather_idx)  # [C, K, D]

    main_bank = F.normalize(main_bank, dim=-1)
    aux_bank = F.normalize(aux_bank, dim=-1)

    return main_bank, aux_bank, aux_indices, text_features_all, sims