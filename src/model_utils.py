import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundedAddFusion(nn.Module):
    """
    fused = normalize(global + w * evidence)
    where:
        w = max_evidence_weight * sigmoid(raw_weight)
    """

    def __init__(self, feat_dim, max_evidence_weight=0.3, init_evidence_weight=0.1):
        super().__init__()
        if max_evidence_weight <= 0:
            raise ValueError("max_evidence_weight must be positive.")
        if init_evidence_weight <= 0 or init_evidence_weight >= max_evidence_weight:
            raise ValueError("init_evidence_weight must be in (0, max_evidence_weight).")

        self.feat_dim = feat_dim
        self.max_evidence_weight = float(max_evidence_weight)

        init_ratio = init_evidence_weight / max_evidence_weight
        init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
        raw_init = torch.log(torch.tensor(init_ratio / (1.0 - init_ratio), dtype=torch.float32))
        self.raw_evidence_weight = nn.Parameter(raw_init)

    def get_weight(self):
        return self.max_evidence_weight * torch.sigmoid(self.raw_evidence_weight)

    def forward(self, global_feats, evidence_feats):
        global_feats = F.normalize(global_feats, dim=-1)
        evidence_feats = F.normalize(evidence_feats, dim=-1)

        weight = self.get_weight()
        fused = global_feats + weight * evidence_feats
        fused = F.normalize(fused, dim=-1)
        return fused


class FLEXFieldLesionEvidenceExtractor(nn.Module):
    """
    FLEX = Field Lesion Evidence Extractor
    """

    def __init__(
        self,
        feat_dim,
        rank_hidden_dim=256,
        rank_dropout=0.1,
        evidence_hidden_dim=512,
        evidence_dropout=0.1,
        top_r=16,
        pool_tau=0.5,
        dev_weight=0.3,
        rank_weight=0.7,
        spatial_smooth_mu=0.15,
        eps=1e-12
    ):
        super().__init__()
        if top_r <= 0:
            raise ValueError("top_r must be positive.")
        if pool_tau <= 0:
            raise ValueError("pool_tau must be positive.")
        if not (0.0 <= spatial_smooth_mu <= 1.0):
            raise ValueError("spatial_smooth_mu must be in [0, 1].")

        self.feat_dim = feat_dim
        self.top_r = int(top_r)
        self.pool_tau = float(pool_tau)
        self.dev_weight = float(dev_weight)
        self.rank_weight = float(rank_weight)
        self.spatial_smooth_mu = float(spatial_smooth_mu)
        self.eps = eps

        self.rank_head = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1, rank_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(rank_dropout),
            nn.Linear(rank_hidden_dim, 1)
        )

        self.evidence_adapter = nn.Sequential(
            nn.Linear(feat_dim, evidence_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(evidence_dropout),
            nn.Linear(evidence_hidden_dim, feat_dim)
        )

    def _minmax_norm(self, x):
        x_min = x.min(dim=1, keepdim=True).values
        x_max = x.max(dim=1, keepdim=True).values
        return (x - x_min) / (x_max - x_min + self.eps)

    def _spatial_smooth(self, scores):
        if self.spatial_smooth_mu <= 0:
            return scores

        bsz, num_tokens = scores.shape
        grid = int(round(math.sqrt(num_tokens)))
        if grid * grid != num_tokens:
            return scores

        score_map = scores.view(bsz, 1, grid, grid)
        neigh_mean = F.avg_pool2d(score_map, kernel_size=3, stride=1, padding=1)
        smoothed = (1.0 - self.spatial_smooth_mu) * score_map + self.spatial_smooth_mu * neigh_mean
        return smoothed.view(bsz, num_tokens)

    def forward(self, global_feats, patch_tokens):
        global_feats = F.normalize(global_feats, dim=-1)
        patch_tokens = F.normalize(patch_tokens, dim=-1)

        bsz, num_tokens, feat_dim = patch_tokens.shape
        if feat_dim != self.feat_dim:
            raise ValueError(f"Expected feat_dim={self.feat_dim}, got {feat_dim}.")

        cosine = torch.einsum("bd,bnd->bn", global_feats, patch_tokens)
        dev_scores = 1.0 - cosine

        global_expand = global_feats.unsqueeze(1).expand_as(patch_tokens)
        residual = patch_tokens - global_expand
        cosine_feat = cosine.unsqueeze(-1)
        rank_input = torch.cat([patch_tokens, residual, cosine_feat], dim=-1)
        rank_scores = self.rank_head(rank_input).squeeze(-1)

        dev_scores_norm = self._minmax_norm(dev_scores)
        rank_scores_norm = self._minmax_norm(rank_scores)

        evidence_scores = self.dev_weight * dev_scores_norm + self.rank_weight * rank_scores_norm
        evidence_scores = self._spatial_smooth(evidence_scores)

        top_r = min(self.top_r, num_tokens)
        top_idx = torch.topk(evidence_scores, k=top_r, dim=1, largest=True, sorted=True).indices

        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, feat_dim)
        selected_tokens = torch.gather(patch_tokens, dim=1, index=gather_idx)
        selected_scores = torch.gather(evidence_scores, dim=1, index=top_idx)

        attn = F.softmax(selected_scores / self.pool_tau, dim=1).unsqueeze(-1)
        pooled_evidence = torch.sum(attn * selected_tokens, dim=1)

        refined_evidence = pooled_evidence + self.evidence_adapter(pooled_evidence)
        refined_evidence = F.normalize(refined_evidence, dim=-1)

        aux = {
            "selected_top_r": int(top_r),
            "selection_ratio": float(top_r / max(num_tokens, 1)),
            "dev_score_mean": float(dev_scores.mean().detach().cpu().item()),
            "rank_score_mean": float(rank_scores.mean().detach().cpu().item()),
            "evidence_score_mean": float(evidence_scores.mean().detach().cpu().item()),
            "spatial_smooth_mu": float(self.spatial_smooth_mu),
        }
        return refined_evidence, aux


class FLEXCAREModel(nn.Module):
    """
    Variants:
        - baseline
        - flex
        - flex_care

    CARE:
        disagreement = |softmax(global / t) - softmax(evidence / t)|
    """

    def __init__(
        self,
        feat_dim,
        variant="flex_care",
        rank_hidden_dim=256,
        rank_dropout=0.1,
        evidence_hidden_dim=512,
        evidence_dropout=0.1,
        evidence_top_r=16,
        evidence_pool_tau=0.5,
        evidence_dev_weight=0.3,
        evidence_rank_weight=0.7,
        spatial_smooth_mu=0.15,
        max_evidence_weight=0.10,
        init_evidence_weight=0.05,
        care_class_penalty_weight=0.05,
        care_js_temp=1.0,
        care_reliability_beta_class=1.0,
        care_reliability_beta_js=1.0,
        eps=1e-12
    ):
        super().__init__()
        valid_variants = {"baseline", "flex", "flex_care"}
        if variant not in valid_variants:
            raise ValueError(f"variant must be one of {valid_variants}, got {variant}.")

        self.variant = variant
        self.feat_dim = feat_dim
        self.eps = eps

        self.care_class_penalty_weight = float(care_class_penalty_weight)
        self.care_js_temp = float(care_js_temp)
        self.care_reliability_beta_class = float(care_reliability_beta_class)
        self.care_reliability_beta_js = float(care_reliability_beta_js)

        if self.variant != "baseline":
            self.flex = FLEXFieldLesionEvidenceExtractor(
                feat_dim=feat_dim,
                rank_hidden_dim=rank_hidden_dim,
                rank_dropout=rank_dropout,
                evidence_hidden_dim=evidence_hidden_dim,
                evidence_dropout=evidence_dropout,
                top_r=evidence_top_r,
                pool_tau=evidence_pool_tau,
                dev_weight=evidence_dev_weight,
                rank_weight=evidence_rank_weight,
                spatial_smooth_mu=spatial_smooth_mu,
                eps=eps
            )

            self.fusion = BoundedAddFusion(
                feat_dim=feat_dim,
                max_evidence_weight=max_evidence_weight,
                init_evidence_weight=init_evidence_weight
            )
        else:
            self.flex = None
            self.fusion = None

    def get_evidence_weight(self):
        if self.fusion is None:
            return 0.0
        return float(self.fusion.get_weight().detach().cpu().item())

    def _kl_div(self, p, q):
        return torch.sum(p * (torch.log(p + self.eps) - torch.log(q + self.eps)), dim=1)

    def _js_div(self, p, q):
        m = 0.5 * (p + q)
        return 0.5 * self._kl_div(p, m) + 0.5 * self._kl_div(q, m)

    def compute_branch_logits(self, global_feats, evidence_feats, fused_feats, text_bank):
        global_feats = F.normalize(global_feats, dim=-1)
        evidence_feats = F.normalize(evidence_feats, dim=-1)
        fused_feats = F.normalize(fused_feats, dim=-1)
        text_bank = F.normalize(text_bank, dim=-1)

        global_logits = 100.0 * global_feats @ text_bank.t()
        evidence_logits = 100.0 * evidence_feats @ text_bank.t()
        fused_logits = 100.0 * fused_feats @ text_bank.t()
        return global_logits, evidence_logits, fused_logits

    def compute_care_outputs(self, global_logits, evidence_logits, fused_logits):
        p_g = F.softmax(global_logits / self.care_js_temp, dim=1)
        p_e = F.softmax(evidence_logits / self.care_js_temp, dim=1)
        js_div = self._js_div(p_g, p_e)

        disagreement = torch.abs(p_g - p_e)
        calibrated_logits = fused_logits - self.care_class_penalty_weight * disagreement

        pred_idx = calibrated_logits.argmax(dim=1)
        d_top1 = disagreement.gather(1, pred_idx.unsqueeze(1)).squeeze(1)

        reliability = torch.exp(
            -(self.care_reliability_beta_class * d_top1 + self.care_reliability_beta_js * js_div)
        )

        return calibrated_logits, disagreement, js_div, reliability, d_top1

    def forward(self, global_feats, patch_tokens, text_bank):
        global_feats = F.normalize(global_feats, dim=-1)
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        text_bank = F.normalize(text_bank, dim=-1)

        num_patch_tokens = patch_tokens.shape[1]

        if self.variant == "baseline":
            logits = 100.0 * global_feats @ text_bank.t()
            aux = {
                "variant": "baseline",
                "fused_logits": logits,
                "calibrated_logits": logits,
                "global_logits": logits,
                "evidence_logits": logits,
                "disagreement": torch.zeros_like(logits),
                "js_div": torch.zeros(logits.shape[0], device=logits.device),
                "reliability": torch.ones(logits.shape[0], device=logits.device),
                "selected_top_r": 0,
                "selection_ratio": 0.0,
                "evidence_weight": 0.0,
                "num_patch_tokens": int(num_patch_tokens),
            }
            return logits, aux

        evidence_feats, flex_aux = self.flex(global_feats, patch_tokens)
        fused_feats = self.fusion(global_feats, evidence_feats)

        global_logits, evidence_logits, fused_logits = self.compute_branch_logits(
            global_feats=global_feats,
            evidence_feats=evidence_feats,
            fused_feats=fused_feats,
            text_bank=text_bank
        )

        if self.variant == "flex":
            main_logits = fused_logits
            disagreement = torch.zeros_like(fused_logits)
            js_div = torch.zeros(fused_logits.shape[0], device=fused_logits.device)
            reliability = torch.ones(fused_logits.shape[0], device=fused_logits.device)
            calibrated_logits = fused_logits
            d_top1 = torch.zeros(fused_logits.shape[0], device=fused_logits.device)
        else:
            calibrated_logits, disagreement, js_div, reliability, d_top1 = self.compute_care_outputs(
                global_logits=global_logits,
                evidence_logits=evidence_logits,
                fused_logits=fused_logits
            )
            main_logits = calibrated_logits

        aux = {
            "variant": self.variant,
            "global_feats": global_feats,
            "evidence_feats": evidence_feats,
            "fused_feats": fused_feats,
            "global_logits": global_logits,
            "evidence_logits": evidence_logits,
            "fused_logits": fused_logits,
            "calibrated_logits": calibrated_logits,
            "disagreement": disagreement,
            "js_div": js_div,
            "reliability": reliability,
            "evidence_weight": self.fusion.get_weight().detach(),
            "num_patch_tokens": int(num_patch_tokens),
        }
        aux.update(flex_aux)

        if self.variant == "flex_care":
            pred_idx = calibrated_logits.argmax(dim=1, keepdim=True)
            aux["care_disagreement_top1_mean"] = float(
                disagreement.gather(1, pred_idx).mean().detach().cpu().item()
            )
            aux["care_d_top1_mean"] = float(d_top1.mean().detach().cpu().item())
            aux["care_js_div_mean"] = float(js_div.mean().detach().cpu().item())
            aux["care_reliability_mean"] = float(reliability.mean().detach().cpu().item())

        return main_logits, aux


@torch.no_grad()
def apply_star_to_logits(
    calibrated_logits,
    reliability,
    seen_mask,
    risk_weight_reliability=1.0,
    risk_weight_uncertainty=1.0,
    risk_weight_seen_bias=0.25,
    dynamic_seen_suppression_kappa=0.25,
    triage_tau_accept=1.10,
    triage_tau_alert=1.50,
    eps=1e-12
):
    """
    actions: 0=accept, 1=defer, 2=alert
    """
    if triage_tau_accept >= triage_tau_alert:
        raise ValueError("triage_tau_accept must be smaller than triage_tau_alert.")

    probs = F.softmax(calibrated_logits, dim=1)
    num_classes = calibrated_logits.size(1)

    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    norm_entropy = entropy / math.log(max(num_classes, 2))

    seen_mask = seen_mask.to(calibrated_logits.device).bool()
    unseen_mask = ~seen_mask

    if seen_mask.any():
        max_seen_prob = probs[:, seen_mask].max(dim=1).values
    else:
        max_seen_prob = torch.zeros_like(norm_entropy)

    if unseen_mask.any():
        max_unseen_prob = probs[:, unseen_mask].max(dim=1).values
    else:
        max_unseen_prob = torch.zeros_like(norm_entropy)

    seen_bias = F.relu(max_seen_prob - max_unseen_prob)

    risk = (
        risk_weight_reliability * (1.0 - reliability) +
        risk_weight_uncertainty * norm_entropy +
        risk_weight_seen_bias * seen_bias
    )

    gate = torch.clamp(
        (risk - triage_tau_accept) / (triage_tau_alert - triage_tau_accept + eps),
        min=0.0,
        max=1.0
    )

    pred0 = calibrated_logits.argmax(dim=1)
    pred_is_seen = seen_mask[pred0].float()
    effective_gate = gate * pred_is_seen

    dynamic_ts = 1.0 + dynamic_seen_suppression_kappa * effective_gate

    final_logits = calibrated_logits.clone()
    if seen_mask.any():
        final_logits[:, seen_mask] = final_logits[:, seen_mask] / dynamic_ts.unsqueeze(1)

    actions = torch.zeros_like(risk, dtype=torch.long)
    actions = torch.where(risk >= triage_tau_alert, torch.full_like(actions, 2), actions)
    mid_mask = (risk >= triage_tau_accept) & (risk < triage_tau_alert)
    actions = torch.where(mid_mask, torch.full_like(actions, 1), actions)

    aux = {
        "star_risk_mean": float(risk.mean().detach().cpu().item()),
        "star_uncertainty_mean": float(norm_entropy.mean().detach().cpu().item()),
        "star_seen_bias_mean": float(seen_bias.mean().detach().cpu().item()),
        "star_gate_mean": float(gate.mean().detach().cpu().item()),
        "star_effective_gate_mean": float(effective_gate.mean().detach().cpu().item()),
        "star_dynamic_ts_mean": float(dynamic_ts.mean().detach().cpu().item()),
    }

    return final_logits, risk, actions, aux