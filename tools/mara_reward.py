"""Controlled reward ablation helpers; no Base/VLM imports or target selection."""
from collections import OrderedDict, defaultdict
from dataclasses import replace
import hashlib
import math
from types import SimpleNamespace

import torch

from tools.mara_agent import MARAAgent
from tools.mara_diagnostics import stable_id

PROTOCOL = "mara_reward_r_v1"
ARMS = {"R0": .5, "R1": .1, "R2": 0.}
INPUTS = ("base_prob", "base_logits", "layer_maps", "evidence_maps", "global_evidence")
CACHE_KEYS = (*INPUTS, "mask", "labels")


def arm_config(template, arm):
    # All other reward terms, costs, architecture and trajectory settings inherit
    # the same template. The production execution gate is NOT swept in R.
    return replace(template, reward_cls_weight=ARMS[arm], freeze_noop_state=True,
                   gain_safety_margin=0., gain_accept_threshold=0., gain_accept_probability=.5)


def options(epochs, cfg, batch_size=4):
    """Canonical train_mara defaults, explicit/sealed rather than CLI-dependent."""
    return SimpleNamespace(
        epoch=epochs, batch_size=batch_size, grpo_group_size=cfg.group_size,
        grpo_update_epochs=3, gain_warmup_epochs=5, disable_force_refine_warmup=False,
        disable_evidence_bank=cfg.evidence_channels == 0, visual_layers=list(range(cfg.num_layers)),
        mara_map_size=cfg.map_size, w_seg=.7, w_global=.3, w_base_consistency=.05,
        w_grpo=.5, grpo_kl_coef=.01, w_gate_sparse=.001, w_gain_value=.05,
        w_gain_consistency=.05, w_op_aux=.1, entropy_coef=.01, grad_clip=1.,
        mara_lr=1e-4, mara_weight_decay=1e-4, warmup_ratio=.05, min_lr_ratio=.05,
        distributed=False, is_main_process=True,
    )


def initial_agent(cfg, seed):
    # fork_rng avoids consuming the stochastic training stream. Arm configs only
    # differ in a scalar quality coefficient, so every parameter is identical.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return MARAAgent(cfg)


def state_hash(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode())
        value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def split_source(manifest, audit, source, seed, fraction=.2):
    if not 0 < fraction < 1:
        raise ValueError("Calibration fraction must be in (0,1)")
    included = {r["sample_id"]: r for r in audit["rows"] if r["included"]}
    source_paths = {r["image_path"] for r in manifest if r["dataset"] == source}
    target_paths = {r["image_path"] for r in manifest if r["dataset"] != source}
    if source_paths & target_paths:
        raise ValueError("Source/target image paths overlap; refusing data leakage")
    strata = defaultdict(list)
    targets = []
    for record in manifest:
        if record["id"] not in included:
            continue
        if record["dataset"] == source:
            strata[(record["category"], included[record["id"]]["image_label"])].append(record["id"])
        else:
            targets.append(record["id"])
    train, calibration = [], []
    for key, ids in sorted(strata.items()):
        if len(ids) < 2:
            raise ValueError(f"Need >=2 source samples per category/label stratum: {key}")
        ids.sort(key=lambda value: stable_id(seed, "source_split", value))
        count = min(len(ids)-1, max(1, round(len(ids)*fraction)))
        calibration.extend(ids[:count])
        train.extend(ids[count:])
    if not train or not calibration or not targets:
        raise ValueError("Need nonempty source train/calibration and target evaluation sets")
    assert not (set(train) & set(calibration) or set(train) & set(targets))
    return dict(source=source, split_seed=seed, calibration_fraction=fraction,
                train_ids=sorted(train), calibration_ids=sorted(calibration), target_ids=sorted(targets),
                note="MARA-only source split; the frozen Base may have seen source calibration images")


def job_list(seeds):
    return [dict(id=f"{arm}_seed{seed}", arm=arm, seed=seed) for seed in seeds for arm in ARMS]


def scheduler_for(optimizer, epochs, batches, args):
    total = max(1, epochs*batches*args.grpo_update_epochs)
    warmup = max(1, int(total*args.warmup_ratio))
    def scale(step):
        if step < warmup:
            return float(step)/warmup
        progress = (step-warmup)/max(1, total-warmup)
        return args.min_lr_ratio+(1-args.min_lr_ratio)*.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class EvidenceStore:
    """Bounded CPU/GPU LRU, immutable tensors. Cache sizing never changes batches."""
    def __init__(self, loader, device, cpu_gb=4., gpu_gb=12.):
        self.loader, self.device = loader, torch.device(device)
        self.cpu, self.gpu = OrderedDict(), OrderedDict()
        self.cpu_bytes = self.gpu_bytes = 0
        self.cpu_cap = int(cpu_gb*2**30)
        self.gpu_cap = 0
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            self.gpu_cap = int(min(gpu_gb*2**30, max(0, free-8*2**30)))

    @staticmethod
    def size(cache):
        return sum(t.numel()*t.element_size() for t in cache.values() if torch.is_tensor(t))

    def _put(self, pool, key, cache, gpu=False):
        amount = self.size(cache)
        attr, cap = ("gpu_bytes", self.gpu_cap) if gpu else ("cpu_bytes", self.cpu_cap)
        if amount > cap:
            return
        used = getattr(self, attr)
        while pool and used+amount > cap:
            _, old = pool.popitem(last=False)
            used -= self.size(old)
        pool[key] = cache
        setattr(self, attr, used+amount)

    def get(self, record):
        key = record["id"]
        if key in self.gpu:
            self.gpu.move_to_end(key)
            return self.gpu[key]
        if key in self.cpu:
            self.cpu.move_to_end(key)
            cache = self.cpu[key]
        else:
            original = self.loader(record)
            cache = {k: original[k].detach() for k in CACHE_KEYS if original.get(k) is not None}
            self._put(self.cpu, key, cache)
        if self.gpu_cap:
            # Evict BEFORE allocation so cached inputs don't cause a false OOM.
            amount = self.size(cache)
            while self.gpu and self.gpu_bytes+amount > self.gpu_cap:
                _, old = self.gpu.popitem(last=False)
                self.gpu_bytes -= self.size(old)
                del old
            cache = {k: v.to(self.device) for k, v in cache.items()}
            self._put(self.gpu, key, cache, gpu=True)
        return cache

    def batch(self, records):
        caches = [self.get(r) for r in records]
        return {k: torch.cat([c[k] for c in caches], 0).to(self.device)
                for k in CACHE_KEYS if k in caches[0]}

    def clear_gpu(self):
        self.gpu.clear()
        self.gpu_bytes = 0
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class CachedBatches:
    def __init__(self, store, records, batch_size, seed, epoch):
        self.store, self.records, self.batch_size = store, records, batch_size
        generator = torch.Generator().manual_seed(seed+1000003*epoch)
        self.order = torch.randperm(len(records), generator=generator).tolist()

    def __len__(self):
        return math.ceil(len(self.records)/self.batch_size)

    def __iter__(self):
        for offset in range(0, len(self.order), self.batch_size):
            batch = self.store.batch([self.records[i] for i in self.order[offset:offset+self.batch_size]])
            yield {"_mara_cached": {k: batch[k] for k in INPUTS if k in batch},
                   "mask": batch["mask"], "is_anomaly": batch["labels"]}
