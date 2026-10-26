"""Visual-only extra caches and a narrowly audited repair of release 2f826ad.

H training and v1 reference features are unchanged. No checkpoint is rewritten.
"""
from pathlib import Path

import numpy as np

from tools.vlm_review import atomic_json, digest, file_digest, load_json

EXTRA_LAYOUT = "visual_teacher_no_rematching_v1"
VISUAL_FIELDS = ("query", "prior", "base")
OLD_RELEASE_CODE = {
    "alignment_stages_experiment.py": "18d2a8df28c3cb1c24e993f0d47cdbeb79364b105abb769409241710b4743317",
    "tools/alignment_stages.py": "af943fee5237889727cf373760f60799e29a0b13ca7c16827c264b5596473fe6",
    "run_exp_alignment_stages.sh": "b7303833621323a7ac5ed20dda2ed0db8a153368a397866ab01e8dba1d5ffde3",
}
REPAIR_FILE = "export_repair.json"


def visual_view(view, project):
    raw = view["raw"]
    side = int(raw.shape[1] ** .5)
    if side * side != raw.shape[1]:
        raise ValueError("Non-square visual token grid")
    query = (raw @ project).permute(0, 2, 1).reshape(raw.shape[0], -1, side, side)
    return dict(query=query, prior=view["prior"], base=view["base"])


def visual_parity(low, high, old):
    """Check only quantities consumed by T, against sealed FP16/FP32 v1 data.

    Keep the old 1e-3 visual / 1e-5 Base numerical budgets; add a per-value
    FP16 half-ULP allowance instead of globally relaxing residual tolerances.
    All six fields are diagnosed before the caller decides whether to stop.
    """
    report = dict(layout=EXTRA_LAYOUT, passed=True, fields={},
                  reference_policy="R reads sealed v1; reference matching is not recomputed by T export")
    for prefix, view in (("global", low), ("high", high)):
        for field in VISUAL_FIELDS:
            key = prefix + "_" + field
            new = view[field].detach().float().cpu().numpy()
            cached = old[key]
            entry = dict(new_shape=list(new.shape), cached_shape=list(cached.shape), cached_dtype=str(cached.dtype))
            report["fields"][key] = entry
            if new.shape != cached.shape or not np.isfinite(new).all() or not np.isfinite(cached).all():
                entry.update(passed=False, reason="shape or nonfinite data")
                report["passed"] = False
                continue
            delta = np.abs(new.astype(np.float64) - cached.astype(np.float64))
            # frexp avoids overflowing nextafter at the largest finite half.
            half_ulp = np.zeros_like(delta)
            if cached.dtype == np.float16:
                exponent = np.frexp(np.abs(cached.astype(np.float64)))[1]
                half_ulp = np.where(cached == 0, 2. ** -25,
                                    np.maximum(np.exp2(exponent - 12), 2. ** -25))
            tolerance = (1e-5 if field == "base" else 1e-3) + half_ulp
            passed = bool(np.all(delta <= tolerance))
            entry.update(passed=passed, max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                         p99_abs=float(np.quantile(delta, .99)), rmse=float(np.sqrt(np.mean(delta ** 2))),
                         max_half_ulp=float(half_ulp.max()), failed_elements=int((delta > tolerance).sum()))
            report["passed"] &= passed
    return report


def verify_execution(root, cfg, actual_code):
    """Accept current code, or this exact old release with an explicit repair receipt."""
    if cfg["new_code"] == actual_code:
        return False
    path = Path(root) / REPAIR_FILE
    if not path.exists():
        raise ValueError("Code changed: use NEW WORK_DIR, or REPAIR_EXPORT=1 for interrupted 2f826ad H-only runs")
    receipt = load_json(path)
    expected = dict(original_config_sha=file_digest(Path(root) / "config.json"),
                    original_fingerprint=digest(cfg), original_code=OLD_RELEASE_CODE,
                    execution_code=actual_code, layout=EXTRA_LAYOUT)
    if (cfg["new_code"] != OLD_RELEASE_CODE or receipt.get("status") != "complete"
            or any(receipt.get(key) != value for key, value in expected.items())
            or actual_code.get("tools/alignment_stages.py") != OLD_RELEASE_CODE["tools/alignment_stages.py"]):
        raise ValueError("Invalid/stale export repair receipt; no code identity checks were bypassed")
    return True


def repair_interrupted_export(root, actual_code):
    """Archive unsealed extras, retain the immutable config and byte-identical H weights.

    Refuse already-sealed/T/R/evaluated runs, unknown code and symlinked paths.
    A prepared receipt makes the one-directory rename safely restartable.
    """
    root = Path(root).resolve()
    config_path = root / "config.json"
    if not config_path.exists():
        print("Export repair: no previous config; prepare a fresh run", flush=True)
        return
    cfg = load_json(config_path)
    if cfg["new_code"] == actual_code:
        print("Export repair: current configuration, nothing to migrate", flush=True)
        return
    path = root / REPAIR_FILE
    if path.exists() and load_json(path).get("status") == "complete":
        verify_execution(root, cfg, actual_code)
        print("Export repair already complete; preserving resumed work", flush=True)
        return
    if (cfg.get("protocol") != "alignment_three_stage_v2" or cfg["new_code"] != OLD_RELEASE_CODE
            or actual_code.get("tools/alignment_stages.py") != OLD_RELEASE_CODE["tools/alignment_stages.py"]):
        raise ValueError("Repair only supports the unmodified 2f826ad release with unchanged H model/loss code")
    for name in ("extra_source_seal.json", "extra_eval_seal.json", "results", "reports", "distance_stats.json"):
        if (root / name).exists():
            raise ValueError(f"Repair refuses sealed or later-stage work: {name}; use NEW WORK_DIR")
    runs = root / "runs"
    preserved = {}
    for seed in runs.glob("seed*") if runs.exists() else ():
        if seed.is_symlink() or any(p.name.startswith("reference_") for p in seed.iterdir()):
            raise ValueError("Repair refuses symlinks or existing reference runs")
        heads = seed / "heads"
        for mode in heads.iterdir() if heads.exists() else ():
            if mode.name not in ("H1", "H2", "H3", "H1_aux0") or mode.is_symlink():
                raise ValueError("Repair refuses existing T training or unknown head directories")
            for checkpoint in mode.glob("*.pt"):
                if checkpoint.is_symlink():
                    raise ValueError("Repair refuses symlinked checkpoints")
                preserved[str(checkpoint.relative_to(root))] = file_digest(checkpoint)
    extra, archive = root / "extra", root / "recovery" / "extra_before_visualfix1"
    if extra.is_symlink() or (root / "recovery").is_symlink():
        raise ValueError("Repair refuses symlinked extra/recovery directories")
    if extra.exists() and not extra.is_dir():
        raise ValueError("Expected extra to be a directory")
    receipt = dict(original_config_sha=file_digest(config_path), original_fingerprint=digest(cfg),
                   original_code=OLD_RELEASE_CODE, execution_code=actual_code, layout=EXTRA_LAYOUT,
                   preserved_h_checkpoints=preserved, archive=str(archive.relative_to(root)), status="prepared")
    if path.exists():
        previous = load_json(path)
        if previous != receipt:
            raise ValueError("Pending repair identity changed; inspect recovery directory before retrying")
    else:
        if archive.exists():
            raise ValueError("Recovery destination already exists; it will not be overwritten")
        atomic_json(path, receipt)
    # Resolved targets are confined to this experiment. Rename, never recursive delete.
    if root not in extra.resolve().parents or root not in archive.resolve().parents:
        raise ValueError("Repair paths escaped the experiment directory")
    if extra.exists():
        if archive.exists():
            raise ValueError("Both extra and recovery exist; refusing to overwrite either")
        archive.parent.mkdir(parents=True, exist_ok=True)
        extra.rename(archive)
    receipt["status"] = "complete"
    atomic_json(path, receipt)
    print(f"Export repair complete: {len(preserved)} H checkpoints untouched; old extras recoverable at {archive}", flush=True)
