# Small-defect local VLM experiment (v1)

This is a frozen CLIP+DINO experiment, not another Base/MARA training run.
It preserves the old `run_exp_dual_vlm.sh` behavior unless `LOCAL_REVIEW=1`.
Use the dedicated `run_exp_dual_vlm_local.sh` wrapper for the new protocol.
Old results/caches cannot be mixed with this version. Use a NEW work directory.

## Design and implemented scope

1. Export the same frozen Base map and layer-disagreement evidence.
2. Alternate peak/disagreement seeds and grow local connected components at two
   scales with relative thresholds. Keep single pixels/thin components; reject
   oversized or truncated components rather than arbitrarily cropping them into
   a fake small defect. Defaults: at most 6 candidates, each at most 0.5% of the
   detector image, union at most 2%. Broad anomalies remain untouched in Base.
3. Map each candidate back to the original image with independent x/y scales.
   Supported dataset geometry is direct resize + same-size center crop (current
   VisA/MVTec/BTAD/MPDD loaders). If the loader gains padding/cropping/rotation,
   the inverse geometry must be updated before using this exporter.
4. Crop native RGB BEFORE resizing for the VLM: context then clean detail, ONE
   candidate per request. An optional third image is a normal training reference.
   No query paths, GT labels, masks or Base scores enter the prompt. The detail
   crop has a small halo; neither image contains an overlaid anomaly heatmap.
5. Strict decisions: `defect_supported`, `normal_supported`, or
   `insufficient_evidence`; also record visibility, reference comparability and
   short visible evidence. No self-reported confidence score controls correction.
   Unresolved detail, invalid output and abstentions keep Base unchanged.
6. Modify ONLY the disjoint compact support, never the observation rectangle.
   Default enhancement is +0.25 log-odds on supported defects. No suppression by
   default. All other pixels, including saturated probabilities, remain exact.
7. P0: audit the candidate decisions against GT after inference. P1: evaluate
   Base, local enhancement, and a same-mask enhancement control without VLM.
   There is no automatic fitting/tuning on those labels.

This version does NOT implement learned pixel refinement, raw-resolution DINO
re-inference, boundary expansion, RL or distillation. Correction remains on the
512x512 Base map. Native crops improve the VLM evidence, but cannot recover a
defect omitted by the Base candidate mechanism. Check candidate recall before
interpreting the teacher's recall. Area thresholds are fixed starting settings,
not experimentally optimized values or performance guarantees.

Enhance-only cannot create new pixel false negatives at an unchanged threshold,
but it CAN raise false positives and hurt AUROC/PRO/best-F1. The same-mask
control is essential: an increase alone does not prove semantic value.

## Server launch: 2 GPUs, same Base, 480-image VisA pilot

```bash
cd /mnt/qfg/Tate_qfg/SRA-DINO
git pull --ff-only origin main
conda activate qfg_addino

nohup env \
  PYTHONUNBUFFERED=1 \
  BASE_CKPT=/mnt/qfg/Tate_qfg/SRA-DINO/checkpoint/base_visa_hfa3_20260908_225850/ckpt \
  WORK_DIR=./checkpoint/dual_vlm_local_visa_v1 \
  DATASETS=visa MAX_PER_CATEGORY=40 SEED=42 \
  GPU_IDS=0,1 NPROC_PER_NODE=2 EXPORT_BS=4 \
  VLM_PYTHON=/root/miniconda3/envs/sra_vlm/bin/python \
  VLM_MODEL_ID=/mnt/qfg/Tate_qfg/models/Qwen3-VL-8B-Instruct-FP8 \
  CC=/usr/bin/gcc CXX=/usr/bin/g++ \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  VLM_GPU_MEMORY=0.70 VLM_IMAGE_SIZE=512 \
  VLM_MAX_MODEL_LEN=4096 VLM_MAX_TOKENS=512 \
  LOCAL_INPUT=native LOCAL_PROMPT=local \
  LOCAL_CANDIDATES=6 LOCAL_MAX_AREA=0.005 LOCAL_TOTAL_AREA=0.02 \
  CALIBRATION_ALPHA=0.25 NORMAL_REFERENCE=0 INCLUDE_SUPPRESSION=0 \
  RUN_EXPORT=1 RUN_REVIEW=1 RUN_EVAL=1 \
  bash run_exp_dual_vlm_local.sh \
  > nohup_dual_vlm_local_visa_v1.out 2>&1 &
echo $!
```

The checkpoint directory resolves to the largest numeric Base epoch (`14.pth`
when present). Export workers finish/unload before VLM workers start. Each GPU
runs its own FP8 model and disjoint samples; this is sharding, not DDP. More
VLM calls are made than the old one-call-per-image protocol. Existing two conda
environments suffice; no new package requirement was introduced (OpenCV is
already used by the original PRO implementation).

Rerun the same command to resume completed exports and per-candidate reviews.
Do not start another process on the same GPUs while the first is still running.
`flock` protects each work directory. `RUN_EXPORT=0` requires a previously sealed
manifest; `RUN_REVIEW=0` requires ALL candidate decisions (or zero candidates).

```bash
tail -f nohup_dual_vlm_local_visa_v1.out
cat ./checkpoint/dual_vlm_local_visa_v1/results/metric_vlm.txt
```

## P0 input ablations (separate WORK_DIR and nohup log for every variant)

Keep Base, dataset, sample limit, seed and candidate parameters identical:

- A: `LOCAL_INPUT=resized LOCAL_PROMPT=generic NORMAL_REFERENCE=0`
- B: `LOCAL_INPUT=native LOCAL_PROMPT=generic NORMAL_REFERENCE=0`
- C (default): `LOCAL_INPUT=native LOCAL_PROMPT=local NORMAL_REFERENCE=0`
- D: `LOCAL_INPUT=native LOCAL_PROMPT=local NORMAL_REFERENCE=1`

`generic` is a shorter generic industrial prompt with the SAME response schema,
not a byte-for-byte replay of the old multi-ROI prompt. The resized variant
deliberately crops after detector-size resizing. Candidate generation does not
depend on the input/prompt/reference variant, so candidates stay paired.

Run variants sequentially on the two GPUs. The one-run script does not silently
launch a four-experiment sweep. Do not choose settings on the held-out target
test set. Use a genuinely held-out source validation set when available. Current
VisA source pilot may overlap Base training (`train.py` uses source TEST data),
so it is diagnostic, not an independent generalization benchmark.

## Optional normal references and suppression

`NORMAL_REFERENCE=1` reads ONLY normal images in the dataset TRAIN split.
Default pool: 8 deterministically sampled images per category. Candidate context
is compared to the same normalized location in those normal images using a
24x24 RGB appearance descriptor, with mean absolute distance <=0.12. It is only
a conservative appearance prefilter, NOT DINO registration or proven semantic
correspondence. For changing object poses, references may be poor; the VLM must
also judge `reference_match`. No match means no reference image is supplied.
The query itself and byte-identical copies are excluded. Reference files have
SHA256 provenance. Missing normal TRAIN data fails clearly, rather than borrowing
test images. This variant uses target normal training data when run on a target
dataset, so report it separately from a zero-target-data setting.

`INCLUDE_SUPPRESSION=1` additionally reports `suppress`, `bidirectional` and
`control_suppress` modes. Actual suppression requires all of:

- valid `normal_supported`, sufficient visibility;
- an attached reference judged `matched`;
- candidate Base maximum strictly below `CONFLICT_THRESHOLD` (default 0.7).

`SUPPRESS_ALPHA=0.1` is separate from enhancement strength. No reference means
actual suppression stays disabled, even if the extra evaluation modes are on.
The control suppresses all candidate masks without asking VLM and has no safety
veto: it is a diagnostic, not the recommended deployment mode.

## Outputs and exact diagnostic definitions

- `results/metric_vlm.txt`, `metrics.csv`, `metrics.json`: per-category and macro
  PRO/P-AUC/I-AUC/best pixel F1 for each enabled mode. Image scores are map maxima.
- `results/candidate_decisions.csv`: per-candidate verdict, reason, visibility,
  reference status, proposed action, crop paths and GT-positive status, joined
  ONLY during evaluation. Proposed action is NOT necessarily applied (normal
  judgments are ignored by the default enhance-only mode).
- `results/diagnostics.csv`: aggregate candidate coverage, confusion counts,
  abstentions, modification area, changed GT pixels, removed pixel false
  positives and new pixel false negatives at FIXED 0.5.
- `local_crops/`: clean context/detail/reference crops for inspection.
- `supports/`: compact masks aligned to the Base map, in `.npz` format.
- `local_reviews/`: atomic per-candidate JSON with original model response.

GT-positive candidate means ANY defect pixel inside its compact support. TP/FP/
TN/FN count only decisive judgments; `defect_candidate_recall` divides TP by ALL
GT-positive candidates, including abstentions. Do not confuse this with pixel
recall. `normal_verdict_defect_fraction` measures how often a normal verdict
actually contains a defect. `candidate_pixel_recall` is pixel-weighted per
category (unlike the old mean-per-anomalous-image field). Undefined ratios are
null/empty, not zero. Pixel/error counts here are NOT the old image-level counts.

GT connected components: small <=0.1% of image area, medium <=1%, otherwise
large. A candidate hit means >=10% of component pixels are inside compact
supports; a prediction hit means >=10% exceed the fixed score threshold 0.5.
Counts `*_count`, `*_candidate_hit`, `*_base_hit`, `*_updated_hit`, `*_new_miss`
are diagnostic, not official PRO or proof of boundary accuracy. Tiny GT
components are retained. Labels never affect proposal/crop/decision generation.

Do not interpret `invalid=0` as correct decisions. If no eligible candidates
exist, VLM is skipped and all outputs equal Base; `no_candidate_images` makes
this explicit. No improvement is promised: compare candidate recall, teacher
errors, small-component hits and the no-VLM control before adding complexity.

## Diagnose near-total abstention before changing the detector

Use `run_exp_vlm_diagnose.sh` to reuse this sealed export for paired original /
neutral-prompt / larger-minimum-pixel-budget / combined reviews. It records
input geometry, processed images, processor token probes, raw answers and
pixel effects in a separate directory, without Base/MARA training. See
[the diagnostic protocol and nohup command](vlm_diagnosis.md).
