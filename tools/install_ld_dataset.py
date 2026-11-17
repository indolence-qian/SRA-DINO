"""Install only the LD loader and registration into an existing SRA-DINO repo.

Run the copy shipped in ld_adapter with --repo /path/to/SRA-DINO.
Existing dataset roots and evaluation code are preserved.
"""
import argparse
import ast
from pathlib import Path
import shutil


REGISTRATION = '''

# BEGIN LD_DATASET_ADAPTER
import os as _ld_os
from Datasets import ld as _ld_dataset
DATASET_REGISTRY["ld"] = (
    _ld_dataset.Dataset, _ld_dataset.DatasetSplit,
    _ld_os.environ.get("LD_DATA_ROOT", "/mnt/qfg/Tate_qfg/data/LD_AD_20261116"),
)
DATASET_CLASSES["ld"] = set(_ld_dataset.CLASSNAMES)
# END LD_DATASET_ADAPTER
'''


def registered(text):
    if "# BEGIN LD_DATASET_ADAPTER" in text:
        return True
    keys = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("DATASET_REGISTRY", "DATASET_CLASSES"):
                    keys[target.id] = {key.value for key in node.value.keys if isinstance(key, ast.Constant)}
    if set(keys) != {"DATASET_REGISTRY", "DATASET_CLASSES"}:
        raise ValueError("Expected top-level DATASET_REGISTRY and DATASET_CLASSES dictionaries")
    present = ["ld" in values for values in keys.values()]
    if any(present) and not all(present):
        raise ValueError("Incomplete existing LD registration; review Datasets/__init__.py")
    return all(present)


def install(repo, bundle):
    repo, bundle = Path(repo).resolve(), Path(bundle).resolve()
    init = repo / "Datasets" / "__init__.py"
    original = init.read_text(encoding="utf-8-sig")
    updated = original if registered(original) else original.rstrip() + REGISTRATION
    ast.parse(updated)
    files = [(bundle / "ld.py", repo / "Datasets" / "ld.py")]
    for name in ("prepare_ld_dataset.py", "check_ld_dataset.py"):
        files.append((bundle / name, repo / "tools" / name))
    # Check every destination before making any changes.
    for source, destination in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise FileExistsError(f"Existing adapter differs; review before replacing: {destination}")
    if updated != original:
        backup = init.with_name("__init__.py.before_ld.bak")
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}")
        shutil.copyfile(init, backup)
    for source, destination in files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination:
            shutil.copyfile(source, destination)
    if updated != original:
        init.write_text(updated, encoding="utf-8", newline="\n")
    print(f"Installed dataset 'ld' in {repo}")
    print("Default root: /mnt/qfg/Tate_qfg/data/LD_AD_20261116; override with LD_DATA_ROOT")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    install(args.repo, args.bundle)
