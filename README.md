
# PlantWild FLEX-CARE-STAR

This project focuses on **generalized zero-shot learning (GZSL) for plant disease recognition** based on CLIP under a single-prompt setting.

To improve open-world plant disease recognition under field conditions, the framework is built around the following three modules:

- **Module 1: FLEX (Field Lesion Evidence Extractor)**
- **Module 2: CARE (Calibration-Aware Reliability Estimator)**
- **Module 3: STAR (Selective Triage with Adaptive Reweighting)**

In particular:

- **FLEX** extracts lesion-related local evidence from patch tokens and fuses it with global image features;
- **CARE** calibrates fused logits using the disagreement between the global branch and the evidence branch, while also producing a reliability signal;
- **STAR** is applied only during evaluation to perform reliability-aware and uncertainty-aware selective triage, helping alleviate seen-class bias and improve seen/unseen balance.

---

## 1. Repository Structure

```text
plantwild_flex_care_star/
├─ CLIP-main/
│  └─ ...                                 # Official CLIP source code; only model.py is modified
├─ configs/
│  ├─ flex_care_train.yaml
│  └─ flex_care_star_eval.yaml
├─ prompts/
│  └─ plantwild_single_prompt.json
├─ src/
│  ├─ dataset.py
│  ├─ model_utils.py
│  ├─ prompt_utils.py
│  └─ utils.py
├─ train_flex_care.py
├─ eval_flex_care_star.py
├─ README.md
├─ requirements.txt
└─ .gitignore
````

The components are summarized as follows:

* `CLIP-main/`: The CLIP source directory downloaded from the official GitHub repository. The original folder structure is preserved, and only `model.py` is modified as required by this project.
* `configs/`: Stores the training and evaluation configurations used in the main experiments.
* `prompts/`: Stores prompt files. This project currently uses a single-prompt setting.
* `src/dataset.py`: Dataset loading, class split construction, and GZSL data organization.
* `src/model_utils.py`: Core implementation of FLEX, CARE, and STAR-related logic.
* `src/prompt_utils.py`: Prompt processing and text-feature-related utilities.
* `src/utils.py`: General utility functions.
* `train_flex_care.py`: Training script for FLEX + CARE.
* `eval_flex_care_star.py`: Evaluation script that loads a trained checkpoint and optionally applies STAR during inference.

---

## 2. Environment and Installation

The reference environment for this project is:

* Python 3.10
* PyTorch 2.5.1
* torchvision 0.20.1
* CUDA 12.1

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## 3. Modified CLIP Backbone

This project uses the `CLIP-main` directory downloaded from the official GitHub repository and keeps its original folder name.

To support local evidence modeling, we modify `model.py` so that the ViT image encoder supports:

* `return_patch_tokens=True`

This allows the model to return both:

* global image features
* patch-level token representations

These patch-level tokens are required by **FLEX**.

Therefore, the modified `model.py` in this project should **not** be replaced with the unmodified original CLIP version; otherwise, the related code will not run properly.

---

## 4. Data Preparation

The expected dataset directory structure is:

```text
datasets/plantwild/
├─ base_train/
├─ base_test/
└─ new_test/
```

where:

* `base_train/`: seen-class training set
* `base_test/`: seen-class test set
* `new_test/`: unseen-class test set

Please organize the dataset in the above format before running the training or evaluation scripts.

---

## 5. Prompt File

This project currently uses a **single-prompt** setting with the following prompt file:

```text
prompts/plantwild_single_prompt.json
```

In this setting:

* each class uses one prompt
* therefore, `expected_num_prompts=1`

If multi-prompt experiments are added later, corresponding JSON files can be placed in `prompts/` and the related arguments can be adjusted accordingly.

---

## 6. Config Files

The `configs/` directory contains two configuration files for the main experiments:

* `flex_care_train.yaml`: training configuration
* `flex_care_star_eval.yaml`: evaluation configuration

These files record the final parameter settings used in the main experiments for reproducibility and management. Even if the scripts are run mainly through command-line arguments, the YAML files still serve as the reference configurations.

---

## 7. Method Overview

### Module 1: FLEX (Field Lesion Evidence Extractor)

**FLEX** is used to extract lesion-related local evidence from patch tokens and fuse it with global semantic features.

Its main idea is:

1. extract global features and patch-level tokens from the modified CLIP image encoder;
2. compute global-guided local deviation scores and learnable ranking scores for patch tokens;
3. select the most informative local evidence tokens;
4. aggregate the selected local evidence features;
5. fuse the enhanced evidence representation with the global representation.

FLEX serves as the main recognition enhancement module in the framework.

---

### Module 2: CARE (Calibration-Aware Reliability Estimator)

**CARE** calibrates classification logits based on the disagreement between the global branch and the evidence branch in the text feature space.

In the current implementation, CARE uses the probability-gap formulation to:

1. compute branch probability distributions from the global logits and evidence logits;
2. measure their disagreement;
3. penalize the fused logits using this disagreement;
4. estimate a reliability score using both class-level disagreement and Jensen-Shannon divergence.

CARE is designed not only to adjust logits, but also to provide a usable reliability signal for downstream decision-making.

---

### Module 3: STAR (Selective Triage with Adaptive Reweighting)

**STAR** is applied only during evaluation.

Its purpose is to use reliability, predictive uncertainty, and seen-class bias to perform selective triage and adaptive suppression of seen-class logits.

More specifically, STAR:

1. computes a risk score from reliability, entropy-based uncertainty, and seen-bias terms;
2. maps the risk score into an adaptive gate;
3. suppresses seen-class logits only when the current prediction belongs to a seen class;
4. produces selective triage actions such as accept, defer, and alert for analysis.

STAR is used as an evaluation-time decision layer to improve seen/unseen balance in GZSL.

---

## 8. Training

The training script is:

```text
train_flex_care.py
```

This script is used to train:

* **FLEX**
* **CARE**

An example training command is:

```bash
python train_flex_care.py \
  --variant flex_care \
  --data_root ./datasets/plantwild \
  --prompt_json ./prompts/plantwild_single_prompt.json \
  --model_name ViT-B/16 \
  --batch_size 8 \
  --num_workers 4 \
  --epochs 50 \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --rank_hidden_dim 256 \
  --rank_dropout 0.1 \
  --evidence_hidden_dim 512 \
  --evidence_dropout 0.1 \
  --evidence_top_r 16 \
  --evidence_pool_tau 0.5 \
  --evidence_dev_weight 0.3 \
  --evidence_rank_weight 0.7 \
  --spatial_smooth_mu 0.15 \
  --max_evidence_weight 0.10 \
  --init_evidence_weight 0.05 \
  --care_class_penalty_weight 0.75 \
  --care_js_temp 1.0 \
  --care_reliability_beta_class 1.0 \
  --care_reliability_beta_js 1.0 \
  --expected_num_prompts 1 \
  --output_dir ./outputs/results/flex_care
```

The main outputs during training typically include:

* the best checkpoint
* training logs
* evaluation results across epochs

Please refer to the actual saving logic in the code for the exact output files.

---

## 9. Evaluation

The evaluation script is:

```text
eval_flex_care_star.py
```

This script is used to:

1. load a trained checkpoint;
2. perform inference with FLEX + CARE;
3. optionally apply **STAR** during evaluation to obtain the final GZSL results.

An example evaluation command is:

```bash
python eval_flex_care_star.py \
  --variant flex_care \
  --data_root ./datasets/plantwild \
  --prompt_json ./prompts/plantwild_single_prompt.json \
  --model_name ViT-B/16 \
  --batch_size 8 \
  --num_workers 4 \
  --rank_hidden_dim 256 \
  --rank_dropout 0.1 \
  --evidence_hidden_dim 512 \
  --evidence_dropout 0.1 \
  --evidence_top_r 16 \
  --evidence_pool_tau 0.5 \
  --evidence_dev_weight 0.3 \
  --evidence_rank_weight 0.7 \
  --spatial_smooth_mu 0.15 \
  --max_evidence_weight 0.10 \
  --init_evidence_weight 0.05 \
  --care_class_penalty_weight 0.75 \
  --care_js_temp 1.0 \
  --care_reliability_beta_class 1.0 \
  --care_reliability_beta_js 1.0 \
  --star_risk_weight_reliability 1.0 \
  --star_risk_weight_uncertainty 1.0 \
  --star_risk_weight_seen_bias 0.0 \
  --star_dynamic_seen_suppression_kappa 0.15 \
  --star_triage_tau_accept 1.1 \
  --star_triage_tau_alert 1.5 \
  --expected_num_prompts 1 \
  --resume_ckpt ./outputs/results/flex_care/best_h_flex_care.pt \
  --output_dir ./outputs/results/flex_care_star_eval
```

If you want to evaluate **FLEX + CARE only** without STAR, you can additionally set:

```bash
--disable_star
```

Notes:

* STAR is used only during evaluation;
* the evaluation script requires a trained checkpoint in advance;
* `resume_ckpt` should point to the checkpoint saved during training.

---

## 10. Key Arguments

Some important arguments are listed below:

### FLEX-related

* `evidence_top_r`: number of selected local evidence patches
* `evidence_pool_tau`: temperature parameter for soft pooling during evidence aggregation
* `evidence_dev_weight`: weight of the global-guided deviation branch in FLEX
* `evidence_rank_weight`: weight of the learnable ranking branch in FLEX
* `max_evidence_weight`: maximum fusion weight for local evidence
* `init_evidence_weight`: initial fusion weight for local evidence
* `spatial_smooth_mu`: spatial smoothing factor for evidence scores

### CARE-related

* `care_class_penalty_weight`: calibration penalty weight
* `care_js_temp`: temperature used when computing branch probabilities for CARE
* `care_reliability_beta_class`: class-level disagreement weight in the reliability term
* `care_reliability_beta_js`: Jensen-Shannon divergence weight in the reliability term

### STAR-related

* `star_risk_weight_reliability`: risk weight for reliability
* `star_risk_weight_uncertainty`: risk weight for uncertainty
* `star_risk_weight_seen_bias`: risk weight for seen-class bias
* `star_dynamic_seen_suppression_kappa`: adaptive suppression strength for seen logits
* `star_triage_tau_accept`: lower threshold for triage
* `star_triage_tau_alert`: upper threshold for triage

---

## 11. Output Files

Training and evaluation results are saved according to `--output_dir`.

Typical outputs include:

* checkpoints saved during training
* epoch-wise history files
* final evaluation result files
* optional per-sample outputs during evaluation

It is recommended to use different `output_dir` values for different experiments for easier result management.

---

## 12. Acknowledgement and License

This project is developed based on OpenAI CLIP and keeps the original source directory as `CLIP-main/`.

Only `model.py` is modified as required by the proposed method to support patch-level token output.

If you use this project, please also follow the license requirements of the original CLIP project and acknowledge the original source where appropriate.
