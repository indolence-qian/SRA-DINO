#!/usr/bin/env python3
"""Re-evaluate the last cached answer, without calling VLM or changing source."""
import argparse
from pathlib import Path
import shutil
from types import SimpleNamespace

import vlm_diagnose as diag
from tools.local_vlm_protocol import parse_repaired
from tools.vlm_review import load_json, atomic_json, file_digest, digest


def run(source_dir, output_dir):
    source, out = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source == out or source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError("Source and output must be separate directories")
    old = load_json(source / "diagnosis_config.json")
    if old["protocol"] != diag.PROTOCOL:
        raise ValueError("Expected a previous paired diagnostic experiment")
    # Deliberately do not demand the OLD code hash equal NEW code. Instead
    # verify its immutable source assets and every old review/trace fingerprint.
    for relative, expected in old["assets"].items():
        if file_digest(diag.safe_asset(old["source_dir"], relative)) != expected:
            raise ValueError(f"Source asset changed: {relative}")
    cfg = dict(old, parser_policy="repair_v2", code_hashes=diag.code_hashes(),
               reparse_source=str(source), reparse_source_fingerprint=digest(old),
               replay_policy="LAST response only, no VLM calls or best-answer selection")
    snapshots = {}
    payloads = []
    for variant in diag.VARIANTS:
        for record, candidate in diag.entries(old):
            name = f"{variant}/{diag.key(record,candidate)}.json"
            path = diag.safe_asset(source, name)
            payload = diag.validate_review(path, old)
            trace = load_json(diag.safe_asset(path.parent, payload["final_trace"]))
            if trace["raw_response"] != payload["raw_responses"][-1]:
                raise ValueError(f"Trace and raw answer mismatch: {name}")
            snapshots[name] = file_digest(path)
            payloads.append((name, candidate, payload))
    cfg["reparse_review_hashes"] = snapshots
    config_path = out / "diagnosis_config.json"
    if config_path.exists() and load_json(config_path) != cfg:
        raise ValueError("Output settings changed: use a NEW output directory")
    atomic_json(config_path, cfg)
    audit_rows = []
    for name, candidate, original in payloads:
        decision, audit = parse_repaired(original["raw_responses"][-1], candidate["roi_id"], "reference" in candidate)
        for relative in original["trace_hashes"]:
            src = diag.safe_asset(source / Path(name).parent, relative)
            dst = out / Path(name).parent / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        updated = dict(original, fingerprint=digest(cfg), decision=decision, **audit)
        updated["previous_decision"] = original["decision"]
        atomic_json(out / name, updated)
        audit_rows.append(dict(key=original["key"], variant=original["variant"],
                               before_valid=original["decision"]["parse_ok"], after_valid=decision["parse_ok"],
                               before_action=diag.action(original["decision"]), after_action=diag.action(decision),
                               errors=";".join(audit["parse_errors"]), warnings=";".join(audit["parse_warnings"])))
    diag.write_csv(out / "reparse_changes.csv", audit_rows)
    diag.evaluate(SimpleNamespace(work_dir=str(out)))
    print(f"Offline replay complete: {out}; source unchanged; VLM calls=0", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_dir", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    run(args.source_dir, args.output_dir)
