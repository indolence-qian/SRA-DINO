# AD-DINOv3: Enhancing DINOv3 for Zero-Shot Anomaly Detection with Anomaly-Aware Calibration

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2509.14084-b31b1b.svg)](https://arxiv.org/abs/2509.14084)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

![](asset/comparison.png)


## Abstract
Zero-Shot Anomaly Detection (ZSAD) seeks to identify anomalies from arbitrary novel categories, offering a scalable and annotation-efficient solution. Traditionally, most ZSAD works have been based on the CLIP model, which performs anomaly detection by calculating the similarity between visual and text embeddings. Recently, vision foundation models such as DINOv3 have demonstrated strong transferable representation capabilities. In this work, we are the first to adapt DINOv3 for ZSAD. However, this adaptation presents two key challenges: (i) the domain bias between large-scale pretraining data and anomaly detection tasks leads to feature misalignment; and (ii) the inherent bias toward global semantics in pretrained representations often leads to subtle anomalies being misinterpreted as part of the normal foreground objects, rather than being distinguished as abnormal regions. To overcome these challenges, we introduce AD-DINOv3, a novel vision-language multimodal framework designed for ZSAD. Specifically, we formulate anomaly detection as a multimodal contrastive learning problem, where DINOv3 is employed as the visual backbone to extract patch tokens and a CLS token, and the CLIP text encoder provides embeddings for both normal and abnormal prompts. To bridge the domain gap, lightweight adapters are introduced in both modalities, enabling their representations to be recalibrated for the anomaly detection task. Beyond this baseline alignment, we further design an Anomaly-Aware Calibration Module (AACM), which explicitly guides the CLS token to attend to anomalous regions rather than generic foreground semantics, thereby enhancing discriminability. Finally, anomaly localization are achieved by measuring the similarity between adapted visual features and prompt embeddings. Extensive experiments on eight industrial and medical benchmarks demonstrate that AD-DINOv3 consistently matches or surpasses state-of-the-art methods, verifying its effectiveness and broad applicability as a general zero-shot anomaly detection framework.

## Results
![](asset/visualization.png)

## Quick Start 
### 1. Installation  
```bash
git clone https://github.com/Kaisor-Yuan/AD-DINOv3.git
cd AD-DINOv3
conda create -n AD-DINOv3 python=3.10  
conda activate AD-DINOv3  
pip install -r requirements.txt  
```
### 2. Datasets
The datasets evaluated in our paper include MVTec AD, VisA, MPDD, BTAD, ISIC, CVC-ColonDB, CVC-ClinicDB, TN3K. Please organize the datasets as follows:
```
./Data/Industrial_Datasets/
├── MVTecAD/
│ ├── bottle/
│ │ ├── test/
│ │ │ ├── broken_large/
│ │ │ ├── ...
│ │ │ └── good/
│ │ └── ground_truth/
│ │   ├── broken_large/
│ │   ├── ...
│ │   └── contamination/
│ ├── ...
│ └── zipper/
│
├── VisA_20220922/
│ ├── candle/
│ ├── ...
│ └── pipe_fryum/
│
├── MPDD/
│ ├── bracket_black/
│ ├── ...
│ └── tubes/
│
└── BTech_Dataset_transformed/
  ├── 01/
  ├── ...
  └── 03/

./Data/Medical_Datasets/
├── ISIC/
│ └── 01/
├── CVC-ColonDB/
│ └── 01/
├── CVC-ClinicDB/
│ └── 01/
└── TN3K/
  └── 01/
```

Put all the industrial datasets under ``./Data/Industrial_Datasets`` and Medical datasets under ``./Data/Medical_Datasets``. Please see the specific details in ``./Datasets/__init__.py``

### 3. Training & Evaluation
```bash
# training
python train.py --result_path $save_path --device $device --dataset $dataset
# evaluation
python test.py --result_path $save_path --dataset $dataset
# (Optional) we provide bash script for evaluating all the datasets
bash test.sh
```

### 4. Restored Mainline: CLIP + DINO

`run_exp.sh` now defaults to `BASE_ARCH=clip_dino` and runs the existing
`train_mara_visa.sh` pipeline: original dual-tower base -> MARA -> target tests.
This restores the architecture/entry point, not the exact numerical results of
an old checkpoint. The base trainer, HFA and semantic-anchor defaults are unchanged.
For strict reproduction, reuse the previously evaluated dual-tower checkpoint
and match its data/preprocessing settings. Single-tower checkpoints are rejected.

```bash
# Train the original base, then MARA, then evaluate.
nohup env BASE_ARCH=clip_dino RUN_BASE=1 RUN_MARA=1 RUN_TEST=1 \
  RUN_VLM_CACHE=0 RUN_VLM_DISTILL=0 GPU_IDS=0,1 NPROC_PER_NODE=2 \
  bash run_exp.sh > nohup_clip_dino_mara.out 2>&1 &

# Prefer this for reproducing a known base (replace the checkpoint path).
nohup env BASE_ARCH=clip_dino RUN_BASE=0 RUN_MARA=1 RUN_TEST=1 \
  RUN_VLM_CACHE=0 RUN_VLM_DISTILL=0 \
  BASE_CKPT=./checkpoint/base_visa_hfa3_xxx/ckpt/14.pth \
  GPU_IDS=0,1 NPROC_PER_NODE=2 \
  bash run_exp.sh > nohup_clip_dino_resume.out 2>&1 &
```

The original dual-tower **base trainer is single-GPU**, MARA uses two-GPU DDP,
and evaluation distributes target datasets across GPUs. Do not launch the
original `train.py` with torchrun: it does not implement DDP. Target checkpoint
selection defaults to the final epoch, not the highest target-test score.

Dual-tower **direct VLM ROI review is now available** through the separate
`run_exp_dual_vlm.sh` entry point below. Dual-tower VLM distillation/MARA integration
is still not implemented. Old single-tower VLM flags on `run_exp.sh` fail explicitly.
See [dual-tower VLM feasibility and ablations](docs/dual_tower_vlm_plan.md).

**Small-defect local review (new):** use `bash run_exp_dual_vlm_local.sh` with
the same dual-tower `BASE_CKPT`. It crops context/detail from the original image,
reviews ONE candidate per VLM request, and only changes compact candidate masks.
Default: enhancement only, at most 0.5% image area per candidate / 2% total;
no Base/MARA retraining. Normal TRAIN references and suppression comparisons
are opt-in. The old multi-ROI script stays available as a baseline.
See [the local-review launch command, P0/P1 ablations and diagnostic definitions](docs/local_vlm_experiment.md).

### 5. Dual Tower + VLM ROI Review (frozen-model pilot)

This experiment does NOT retrain Base or the 8B VLM, and bypasses MARA. It first
exports CLIP+DINO maps and four candidate ROIs per sample, exits all dual-tower
workers, runs one image-only FP8 VLM per GPU on disjoint shards, then unloads
the VLM and computes CPU metrics. Continue using the original training conda
environment plus the isolated `sra_vlm` environment; no new model is needed.

```bash
conda activate qfg_addino
nohup env \
  BASE_CKPT=/mnt/qfg/Tate_qfg/SRA-DINO/checkpoint/base_visa_hfa3_20260908_225850/ckpt \
  WORK_DIR=./checkpoint/dual_vlm_visa_pilot_v1 \
  DATASETS=visa MAX_PER_CATEGORY=40 \
  GPU_IDS=0,1 EXPORT_BS=4 \
  VLM_PYTHON=/root/miniconda3/envs/sra_vlm/bin/python \
  VLM_MODEL_ID=/mnt/qfg/Tate_qfg/models/Qwen3-VL-8B-Instruct-FP8 \
  CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  VLM_GPU_MEMORY=0.70 VLM_IMAGE_SIZE=512 VLM_MAX_MODEL_LEN=4096 VLM_MAX_TOKENS=512 \
  VLM_HEATMAP=0 CALIBRATION_ALPHA=0.5 VLM_CONFIDENCE_THRESHOLD=0.8 \
  bash run_exp_dual_vlm.sh > nohup_dual_vlm_visa_pilot_v1.out 2>&1 &
```

- A checkpoint directory selects the highest **numeric** epoch filename and
  prints the resolved path; a MARA or single-tower checkpoint is rejected.
- The default pilot samples up to 40 images per VisA category with a fixed
  seed. It is a development diagnostic, NOT independent validation if these
  images were used to train Base. The sample is not selected by model score.
- After fixing settings using appropriate source data, full target evaluation
  uses the same command with `DATASETS="mvtec btad mpdd" MAX_PER_CATEGORY=0`
  and **a new** `WORK_DIR=./checkpoint/dual_vlm_targets_v1` and log filename.
  Do not tune calibration parameters on those target test results.
- There are no normal-reference inputs. Query/source label-bearing paths and
  GT masks are never placed in the VLM prompt. `VLM_HEATMAP=1` is a separate
  hint ablation and requires a new work directory.
- Confidence is an uncalibrated eligibility filter, NOT a calibrated defect
  probability. Accepted votes modify ROI log-odds by at most `CALIBRATION_ALPHA`
  with tapered edges. Uncertain, low-confidence or invalid reviews retain Base.
- `results/metric_vlm.txt`, `metrics.csv`, and `metrics.json` compare `base`,
  `vlm_image` (only the image score changes), `vlm_region` (ROI map and derived
  image score change), and `control_shrink` (non-VLM ROI suppression).
  Image scores are map maxima, matching `test_mara.py`; map normalization is
  `none`. Best-F1 is a test-set reporting statistic, not a deployable threshold.
- Results also report candidate pixel recall, decisive ROI accuracy/coverage,
  mean query latency, invalid responses and FP/FN counts at a fixed diagnostic
  threshold of 0.5. This is inference-time VLM usage even though results are cached;
  do not report it as a VLM-free deployed detector.
- Rerun the **same command** to resume atomically completed samples. A lock
  prevents simultaneous use of one work directory; changed settings/weights
  are rejected. `RUN_EXPORT=0` skips an already sealed export, `RUN_REVIEW=0`
  skips an already complete review. No missing reviews are silently ignored;
  more than 5% invalid responses fail evaluation by default.
- Runtime settings are frozen in `config.json` before evaluation, with a Base
  SHA256 and local VLM weight size/mtime signature. Do not edit cache files or
  move/change source data during a run. Ensure disk space for full-resolution
  probability maps, query PNGs and evaluation masks.

GPU availability/CC/Triton checks run before export. Failed worker stages stop
the pipeline; no later training stage starts. On interruption, verify the old
workers have exited with `nvidia-smi` before restarting. Real dual-4090 FP8
capacity/throughput must be checked on the server; CPU/mock tests do not establish it.

### 6. Retained Ablation: DINO Single Tower + Qwen3-VL FP8 Teacher

In this explicitly selected ablation, the deployed detector is a language-free DINO single tower. During
training only, two independent `Qwen3-VL-8B-Instruct-FP8` workers inspect the
query, a same-category normal reference, the DINO heatmap, and proposed ROIs.
Their JSON decisions are cached and distilled into a small semantic decision
head. The 8B VLM is then unloaded before DDP distillation and MARA training, so
the final detector does not require vLLM or the VLM weights.

```bash
# One-time CUDA 12.8 VLM environment. Keep it separate from the training env.
conda create -n sra_vlm python=3.10 pip -y
conda activate sra_vlm
python -m pip install -U uv
uv pip install --python "$(which python)" -r requirements-vlm.txt --torch-backend=cu128
python -c "import torch,vllm; print(torch.__version__, torch.version.cuda, vllm.__version__, torch.cuda.is_available())"
conda activate qfg_addino

# Base -> two FP8 cache workers -> semantic distillation -> MARA -> tests.
nohup env GPU_IDS=0,1 NPROC_PER_NODE=2 \
  VLM_PYTHON=/root/miniconda3/envs/sra_vlm/bin/python \
  bash run_exp_dino_single.sh > nohup_dino_qwen3vl_mara.out 2>&1 &
```

The default protocol trains with labeled VisA source images and evaluates on
MVTec AD, BTAD, and MPDD. Do not report VisA as an unseen target in this setup.
Base, semantic-head, and MARA training use the original environment and
two-GPU DDP. VLM caching uses the isolated `sra_vlm` environment and is not
tensor parallel: GPU 0 and GPU 1 each run a complete FP8 model on a disjoint,
resumable data shard. Cross-dataset evaluation assigns datasets across GPUs.

Useful stage controls:

```bash
# Reuse an existing base and VLM cache, then distill/train/test.
nohup env RUN_BASE=0 RUN_VLM_CACHE=0 \
  BASE_CKPT=./checkpoint/dino_single_visa_xxx/ckpt/single_epoch_14.pth \
  VLM_CACHE_PATH=./checkpoint/vlm_teacher_xxx/qwen3_vl_fp8_decisions.jsonl \
  GPU_IDS=0,1 NPROC_PER_NODE=2 \
  bash run_exp_dino_single.sh > nohup_resume_vlm_mara.out 2>&1 &

# Skip the VLM route entirely and retain the original DINO-only baseline.
RUN_VLM_CACHE=0 RUN_VLM_DISTILL=0 GPU_IDS=0,1 NPROC_PER_NODE=2 bash run_exp_dino_single.sh
```

Alternatively, `BASE_ARCH=dino_single bash run_exp.sh` selects this ablation.
`run_exp_dino_single.sh` defaults to 70% GPU allocation per VLM worker. The image-only
teacher disables video profiling and CUDA graphs, permits one request at a time,
and sets `max_num_batched_tokens` to `VLM_MAX_MODEL_LEN` (4096 by default).
`VLM_IMAGE_SIZE` (512 by default) controls input thumbnails and the pixel-area
cap (`VLM_IMAGE_SIZE ** 2`) in both Qwen preprocessing and vLLM profiling.
Eager execution reduces startup overhead but may reduce peak throughput.

If startup reports **negative available KV cache memory**, lowering
`VLM_GPU_MEMORY` makes that budget smaller, not larger. First check the log
shows `video: 0`, `max_num_seqs: 1`, and `enforce_eager: True`. If still needed,
try `VLM_GPU_MEMORY=0.80` only when sufficient memory remains for DINO and no
unrelated jobs occupy the cards. Conversely, an error saying **free memory is
less than desired utilization** requires freeing competing allocations or
lowering the budget. Neither setting guarantees that all workloads fit.

Triton also requires a system C compiler even with eager execution. On
Ubuntu/Debian install `build-essential` (as root or with sudo), and optionally
pass `CC=/usr/bin/gcc CXX=/usr/bin/g++` in the `nohup env` command. This is a
system dependency, not a package in `requirements-vlm.txt`.

A failed cache shard is append-only and resumable by rerunning with the same
`BASE_CKPT`, teacher settings and `VLM_WORK_DIR`. Use a new work directory if
changing teacher inputs after valid decisions have already been cached. The
merged cache must cover at least 95% of source samples before distillation starts.

To run the code, please download the pretrained weights and place them in the specified directories:

| ✅ **Pretrained Model** | 🌐 **Source Link** | 📁 **Destination Path** |
|:-----------------------:|:------------------|:------------------------|
| **OpenCLIP ViT-L-14-336px** | [🔗 OpenCLIP Checkpoints](https://github.com/mlfoundations/open_clip) | `./CLIP/ckpt/` |
| **Meta AI DINOv3 (vitl16_pretrain_lvd)** | [🔗 DINOv3 Official Weights](https://github.com/facebookresearch/dinov3) | `./` *(project root)* |


## Citation
If you use this work, please cite:
```
@article{yuan2025ad,
  title={AD-DINOv3: Enhancing DINOv3 for Zero-Shot Anomaly Detection with Anomaly-Aware Calibration},
  author={Yuan, Jingyi and Ye, Jianxiong and Chen, Wenkang and Gao, Chenqiang},
  journal={arXiv preprint arXiv:2509.14084},
  year={2025}
}
```

⭐ Star this repo if you find it useful!
