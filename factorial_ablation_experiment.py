#!/usr/bin/env python3
"""Controlled 2x2 HR/representation ablation; no distance fallback in any arm.

The previous experiment remains unchanged. All four arms train the same heads;
appearance arms never read reference features, residuals or nearest-neighbor scores.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import datetime
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch

import residual_text_experiment as base
from tools.residual_text import ResidualDensity, ResidualText, density_loss, text_loss, fuse_views, positive_text_fusion
from tools.residual_text_runtime import (FrozenDinoExtractor, GENERIC_PROMPT, TEMPLATES, atomic_npz,
    encode_text_bank, evaluation_mask, export_residuals, normalized_boxes, read_torch, save_torch, source_labels, unchanged)
from tools.vlm_review import atomic_json, digest, file_digest, load_json

REPO = Path(__file__).resolve().parent
PROTOCOL = "hr_representation_factorial_v1"
ARMS = {"A": ("appearance", "G"), "B": ("residual", "G"),
        "C": ("appearance", "HR"), "D": ("residual", "HR")}
MAPS = tuple(f"{arm}_{kind}" for arm in ARMS for kind in ("visual", "full"))
CODE = tuple(dict.fromkeys(("factorial_ablation_experiment.py", "run_exp_factorial_ablation.sh") + base.CODE_FILES))
METRICS = ("I_AUROC", "I_AP", "P_AUROC", "P_AP", "PRO_exact", "F1_best",
           "background_FPR_source_threshold", "pixel_recall_source_threshold")


def immutable(path, value):
    path = Path(path)
    if path.exists() and load_json(path) != value:
        raise ValueError(f"Immutable artifact differs: {path}; use NEW WORK_DIR")
    if not path.exists():
        atomic_json(path, value)


def config(root):
    root = Path(root)
    cfg = load_json(root/"config.json")
    if cfg["protocol"] != PROTOCOL:
        raise ValueError("Wrong experiment; use NEW WORK_DIR")
    for name, sha in cfg["code"].items():
        if file_digest(REPO/name) != sha:
            raise ValueError(f"Code changed: {name}; use NEW WORK_DIR")
    for name in ("manifest", "references"):
        if file_digest(root/(name+".json")) != cfg[name+"_sha"]:
            raise ValueError("Data manifest changed")
    return cfg


def rows(root, partition):
    data = load_json(Path(root)/"manifest.json")
    return [r for r in data if r["partition"] in ("train", "val")] if partition == "source" else [r for r in data if r["partition"] == partition]


def groups(records):
    result = defaultdict(list)
    for row in records:
        result[(row["dataset"], row["category"])].append(row)
    return dict(sorted(result.items()))


def prepare(args, catalog=None):
    root = Path(args.work_dir).resolve()
    layers = [int(v) for v in args.visual_layers.split(",")]
    if (not layers or len(set(layers)) != len(layers) or min(layers) < 0
            or args.image_size < 32 or args.patch_size < 1 or args.image_size % args.patch_size
            or not 2 <= args.feature_dim <= args.input_dim or args.reference_count not in (1,2,4)
            or not .5 < args.tile_fraction < 1 or not 0 < args.high_weight < 1
            or not 0 < args.val_fraction < .5 or not 0 < args.val_fpr < 1
            or not 0 <= args.text_strength <= 1 or args.density_epochs != args.text_epochs
            or not 1 <= args.until_epoch <= args.density_epochs
            or min(args.batch_size,args.micro_batch,args.flow_blocks,args.hidden,args.num_shards,
                   args.match_chunk,args.bank_chunk,args.tokens_per_view,args.calibration_images,
                   args.eval_size,args.selection_size) < 1 or args.workers < 0
            or args.micro_batch > args.batch_size or args.limit_per_category < 0 or args.semantic_annotations
            or not all(np.isfinite(v) and v > 0 for v in (args.density_lr,args.text_lr))):
        raise ValueError("Invalid factorial experiment settings")
    for name in (args.dino_weights,args.clip_weights):
        if not Path(name).is_file():
            raise FileNotFoundError(f"Local pretrained weights required: {name}")
    if not Path(args.dino_repo_dir,"hubconf.py").is_file():
        raise FileNotFoundError("DINO_REPO_DIR must contain hubconf.py")
    queries,candidates = copy.deepcopy(catalog if catalog is not None else base.load_catalog(args))
    categories = sorted({r["category"] for r in queries if r["partition"]=="source"},key=lambda c:digest([args.data_seed,c]))
    if len(categories)<2:
        raise ValueError("Need at least two source classes")
    count=max(1,min(len(categories)-1,round(len(categories)*args.val_fraction)))
    validation=categories[:count]
    for row in queries:
        if row["partition"]=="source":
            row["partition"]="val" if row["category"] in validation else "train"
    seen,ids={},set()
    for row in queries:
        if row["id"] in ids or seen.setdefault(row["rgb_sha"],row["partition"])!=row["partition"]:
            raise ValueError("Duplicate ID or image content across data partitions")
        ids.add(row["id"])
    for partition in ("train","val"):
        if {bool(r["mask_path"]) for r in queries if r["partition"]==partition}!={False,True}:
            raise ValueError("Both source splits need normal and abnormal images")
    if not any(r["partition"]=="eval" for r in queries):
        raise ValueError("No target queries")
    references,reference_classes=[],{}
    for key in groups(queries):
        eligible=sorted([r for r in candidates if (r["dataset"],r["category"])==key],
                        key=lambda r:digest([args.data_seed,"reference",r["rgb_sha"]]))
        chosen,sha=[],set()
        for row in eligible:
            if row["mask_path"] or row["partition"]!="reference" or row["rgb_sha"] in seen or row["rgb_sha"] in sha:
                continue
            if reference_classes.setdefault(row["rgb_sha"],key)!=key:
                raise ValueError("Reference content crosses classes/datasets")
            chosen.append(row); sha.add(row["rgb_sha"])
            if len(chosen)==args.reference_count:
                break
        if len(chosen)!=args.reference_count:
            raise ValueError(f"Need K distinct normal TRAIN references: {key}")
        references.extend(chosen)
    normals=groups([r for r in queries if r["partition"]=="train" and not r["mask_path"]])
    if args.calibration_images<len(normals) or len(normals)!=len(categories)-count or any(len(v)<2 for v in normals.values()):
        raise ValueError("Each source TRAIN class needs >=2 normal queries; CAL must cover every class")
    for values in normals.values():
        values.sort(key=lambda r:digest([args.data_seed,"factorial_cal",r["id"]]))
    calibration=[]
    for i in range(max(map(len,normals.values()))-1):
        for values in normals.values():
            if i<len(values)-1 and len(calibration)<args.calibration_images:
                calibration.append(values[i]["id"])
    ignored={"stage","work_dir","partition","shard_id","device","variant","train_seed","train_seeds",
             "until_epoch","probe","arm","kind","source_root","target_root","data_roots","publish_report"}
    cfg={k:v for k,v in vars(args).items() if k not in ignored}
    cfg.update(protocol=PROTOCOL,layers=layers,val_categories=validation,calibration_ids=calibration,
        dino_weights=str(Path(args.dino_weights).resolve()),clip_weights=str(Path(args.clip_weights).resolve()),
        dino_repo_dir=str(Path(args.dino_repo_dir).resolve()),dino_sha=file_digest(args.dino_weights),
        clip_sha=file_digest(args.clip_weights),code={n:file_digest(REPO/n) for n in CODE},
        backbone_code=digest({str(p.relative_to(args.dino_repo_dir)):file_digest(p) for p in sorted(Path(args.dino_repo_dir).rglob("*.py"))}),
        torch_version=str(torch.__version__),templates=list(TEMPLATES),
        selection="best trained epoch by source class-macro mean(PRO, AP); no distance/epoch-zero fallback",
        text_policy="same frozen generic text delta and fixed fusion strength in every arm",
        timing_note="paired five-view cache export is not individual-arm inference latency")
    root.mkdir(parents=True,exist_ok=True)
    if (root/"config.json").exists():
        # Compare without touching old artifacts, including sidecar provenance.
        expected=dict(cfg,manifest_sha=file_digest(root/"manifest.json"),references_sha=file_digest(root/"references.json"))
        if load_json(root/"manifest.json")!=queries or load_json(root/"references.json")!=references or load_json(root/"config.json")!=expected:
            raise ValueError("Settings/data/code changed; use NEW WORK_DIR")
        config(root); return expected
    immutable(root/"manifest.json",queries); immutable(root/"references.json",references)
    cfg.update(manifest_sha=file_digest(root/"manifest.json"),references_sha=file_digest(root/"references.json"))
    immutable(root/"config.json",cfg)
    grid=args.image_size//args.patch_size
    print(f"Four arms A/B/C/D; held-out source classes={validation}; normal CAL={len(calibration)} excluded from gradient training",flush=True)
    print(f"Paired appearance/residual cache estimate {len(queries)*2*5*len(layers)*args.feature_dim*grid**2*2/2**30:.1f} GiB, plus predictions; reserve sufficient disk",flush=True)
    return cfg


def feature_path(root,row,representation):
    return Path(root)/"features"/representation/(row["id"]+".npz")


@torch.inference_mode()
def appearance_features(extractor,views,cfg):
    side=cfg["image_size"]//cfg["patch_size"]
    return torch.stack([(v@extractor.projection).permute(0,2,1).reshape(len(cfg["layers"]),cfg["feature_dim"],side,side).cpu()
                        for v in views]).numpy().astype(np.float16)


def export(args):
    root,cfg=Path(args.work_dir),config(args.work_dir)
    if not 0<=args.shard_id<cfg["num_shards"]:
        raise ValueError("Invalid shard")
    supports=groups(load_json(root/"references.json"))
    extractor=None
    try:
        for index,(key,queries) in enumerate(groups(rows(root,args.partition)).items()):
            if index%cfg["num_shards"]!=args.shard_id:
                continue
            pending=[r for r in queries if any(not feature_path(root,r,k).with_suffix(".json").exists() for k in ("appearance","residual"))]
            if not pending:
                continue
            if (root/("sealed_"+args.partition+".json")).exists():
                raise ValueError("Cannot regenerate sealed cache")
            if extractor is None:
                code=digest({str(p.relative_to(cfg["dino_repo_dir"])):file_digest(p) for p in sorted(Path(cfg["dino_repo_dir"]).rglob("*.py"))})
                if code!=cfg["backbone_code"]:
                    raise ValueError("DINO code changed")
                extractor=FrozenDinoExtractor(cfg,base.device_for(args,cfg))
            normal_views=[]
            for row in supports[key]:
                unchanged(row)
                with Image.open(row["image_path"]) as image:
                    views,_=extractor.views(image.convert("RGB"))
                normal_views.append([v.cpu() for v in views])
            for row in pending:
                unchanged(row)
                with Image.open(row["image_path"]) as image:
                    views,boxes=extractor.views(image.convert("RGB"))
                common=dict(boxes=normalized_boxes(boxes,row["native_size"]),pixel_boxes=np.asarray(boxes,np.int32))
                for representation in ("appearance","residual"):
                    path=feature_path(root,row,representation)
                    if path.with_suffix(".json").exists():
                        continue
                    if representation=="appearance":
                        arrays=dict(features=appearance_features(extractor,views,cfg),**common)
                        references=[]  # No normal reference data in appearance inputs.
                    else:
                        values=export_residuals(extractor,views,normal_views,cfg)
                        arrays=dict(features=values["residual"],match_index=values["match_index"],**common)
                        references=[r["id"] for r in supports[key]]
                    atomic_npz(path,**arrays)
                    immutable(path.with_suffix(".json"),dict(id=row["id"],fingerprint=digest(cfg),representation=representation,
                                                           references=references,sha=file_digest(path)))
                print(f"export {args.partition} {row['id']}",flush=True)
    finally:
        if extractor is not None:
            extractor.close()


def load_feature(root,cfg,row,representation,verify=True):
    path=feature_path(root,row,representation)
    if verify:
        meta=load_json(path.with_suffix(".json"))
        refs=[r["id"] for r in load_json(Path(root)/"references.json") if (r["dataset"],r["category"])==(row["dataset"],row["category"])] if representation=="residual" else []
        if (meta["fingerprint"]!=digest(cfg) or meta["id"]!=row["id"] or meta["representation"]!=representation
                or meta["references"]!=refs or meta["sha"]!=file_digest(path)):
            raise ValueError("Feature identity/reference/hash mismatch")
    with np.load(path,allow_pickle=False) as archive:
        data={n:archive[n].copy() for n in archive.files}
    side=cfg["image_size"]//cfg["patch_size"]
    expected={"features","boxes","pixel_boxes"}|({"match_index"} if representation=="residual" else set())
    if (set(data)!=expected or data["features"].shape!=(5,len(cfg["layers"]),cfg["feature_dim"],side,side)
            or any(data[n].shape!=(5,4) for n in ("boxes","pixel_boxes"))
            or any(not np.isfinite(v).all() for v in data.values())):
        raise ValueError("Invalid feature cache arrays")
    if representation=="residual":
        indices=data["match_index"]
        if (indices.shape!=(5,len(cfg["layers"]),side,side) or (indices<0).any()
                or (indices[:1]>=cfg["reference_count"]*side**2).any()
                or (indices[1:]>=4*cfg["reference_count"]*side**2).any()):
            raise ValueError("Invalid correspondence indices")
    return data


def seal(args):
    root,cfg=Path(args.work_dir),config(args.work_dir)
    hashes={}
    for row in rows(root,args.partition):
        for representation in ("appearance","residual"):
            load_feature(root,cfg,row,representation)
            path=feature_path(root,row,representation)
            hashes[representation+"/"+row["id"]]=dict(sha=file_digest(path),metadata=file_digest(path.with_suffix(".json")))
    immutable(root/("sealed_"+args.partition+".json"),dict(fingerprint=digest(cfg),features=hashes))


class Features(torch.utils.data.Dataset):
    def __init__(self,root,cfg,records,arm,labels=False):
        self.root,self.cfg,self.rows,self.arm,self.labels=Path(root),cfg,records,arm,labels
        representation,_=ARMS[arm]
        seals={}
        for row in records:
            part="eval" if row["partition"]=="eval" else "source"
            if labels and part=="eval":
                raise ValueError("Target labels forbidden in training/calibration")
            if part not in seals:
                seals[part]=load_json(self.root/("sealed_"+part+".json"))
                if seals[part]["fingerprint"]!=digest(cfg):
                    raise ValueError("Wrong feature seal")
            path=feature_path(root,row,representation)
            value=seals[part]["features"][representation+"/"+row["id"]]
            if value!={"sha":file_digest(path),"metadata":file_digest(path.with_suffix(".json"))}:
                raise ValueError("Feature changed after sealing")
            unchanged(row,masks=labels)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        representation,variant=ARMS[self.arm]
        data=load_feature(self.root,self.cfg,row,representation,verify=False)
        count=1 if variant=="G" else 5
        result=dict(residual=torch.from_numpy(data["features"][:count].astype(np.float32)),
                    boxes=torch.from_numpy(data["boxes"][:count].astype(np.float32)),index=torch.tensor(index))
        # 'residual' is the shared head API's tensor name, not a residual bypass:
        # appearance datasets load ONLY features/appearance, never residual files.
        if self.labels:
            coverage,typed=source_labels(row,data["pixel_boxes"][:count].tolist(),result["residual"].shape[-1],[],[])
            result.update(coverage=coverage,typed=typed)
        return result


def calibration_rows(root,cfg):
    wanted=set(cfg["calibration_ids"])
    return [r for r in rows(root,"train") if r["id"] in wanted]


def fit_stats(args):
    root,cfg=Path(args.work_dir),config(args.work_dir)
    result={}
    for representation,arm in (("appearance","C"),("residual","D")):
        sums=np.zeros((len(cfg["layers"]),2)); counts=sums.copy()
        for item in Features(root,cfg,calibration_rows(root,cfg),arm,True):
            x=item["residual"].numpy().astype(np.float64)
            for view in range(5):
                sums[:,int(view>0)]+=np.square(x[view]).sum(axis=(1,2,3))
                counts[:,int(view>0)]+=x[view].shape[1]*x[view].shape[2]*x[view].shape[3]
        scale=np.sqrt(sums/counts)
        if not np.isfinite(scale).all() or (scale<1e-8).any():
            raise ValueError("Degenerate source normal scale")
        result[representation]=scale.tolist()
    immutable(root/"normal_stats.json",dict(fingerprint=digest(cfg),source_seal=file_digest(root/"sealed_source.json"),
                                           rms=result,calibration_ids=cfg["calibration_ids"]))


def scale_for(root,cfg,arm,device):
    stats=load_json(Path(root)/"normal_stats.json")
    if stats["fingerprint"]!=digest(cfg) or stats["source_seal"]!=file_digest(Path(root)/"sealed_source.json"):
        raise ValueError("Stale normal scale")
    scale=torch.tensor(stats["rms"][ARMS[arm][0]],dtype=torch.float32,device=device)
    if scale.shape!=(len(cfg["layers"]),2) or not torch.isfinite(scale).all() or (scale<=0).any():
        raise ValueError("Invalid normal scale")
    return scale


def text_bank(args):
    root,cfg=Path(args.work_dir),config(args.work_dir)
    path=root/"text_bank.npz"
    if path.exists() and path.with_suffix(".json").exists():
        directions_for(root,cfg,"cpu"); return
    directions,states=encode_text_bank(cfg,[GENERIC_PROMPT],base.device_for(args,cfg))
    atomic_npz(path,directions=directions,paired_states=states)
    immutable(path.with_suffix(".json"),dict(fingerprint=digest(cfg),sha=file_digest(path)))


def directions_for(root,cfg,device):
    path=Path(root)/"text_bank.npz"; meta=load_json(path.with_suffix(".json"))
    if meta!={"fingerprint":digest(cfg),"sha":file_digest(path)}:
        raise ValueError("Text bank changed")
    with np.load(path,allow_pickle=False) as archive:
        return torch.from_numpy(archive["directions"].copy()).float().to(device)


def new_model(root,cfg,kind,seed,device):
    if kind=="density":
        return ResidualDensity(len(cfg["layers"]),cfg["feature_dim"],cfg["hidden"],cfg["flow_blocks"],seed).to(device)
    return ResidualText(len(cfg["layers"]),cfg["feature_dim"],directions_for(root,cfg,device),cfg["hidden"]).to(device)


def horizon(args,cfg):
    if not 1<=args.until_epoch<=cfg["density_epochs"]:
        raise ValueError("Invalid training horizon")
    return args.until_epoch


def head_dir(root,seed,arm,kind):
    return Path(root)/"heads"/f"seed{seed}"/arm/kind


def training_identity(root,cfg,arm,kind,seed):
    result=dict(fingerprint=digest(cfg),arm=arm,kind=kind,seed=seed,
                source_seal=file_digest(Path(root)/"sealed_source.json"),stats=file_digest(Path(root)/"normal_stats.json"))
    if kind=="text":
        result["text_bank"]=file_digest(Path(root)/"text_bank.npz")
    return result


def quality(point):
    return sum(point[n] if point[n] is not None else -1 for n in ("PRO_exact","P_AP"))/2


def train(args):
    root,cfg=Path(args.work_dir),config(args.work_dir)
    device=base.device_for(args,cfg); until=horizon(args,cfg)
    arm,kind,seed=args.arm,args.kind,args.train_seed
    variant=ARMS[arm][1]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model=new_model(root,cfg,kind,seed,device)
    cal=Features(root,cfg,calibration_rows(root,cfg),arm,True)
    cal_ids=set(cfg["calibration_ids"])
    data=Features(root,cfg,[r for r in rows(root,"train") if r["id"] not in cal_ids],arm,True)
    val=Features(root,cfg,rows(root,"val"),arm,True)
    scale=scale_for(root,cfg,arm,device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg[kind+"_lr"],weight_decay=.01)
    steps=int(np.ceil(len(data)/cfg["batch_size"]))*cfg[kind+"_epochs"]
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:.05+.95*(1+np.cos(np.pi*min(s,steps)/steps))/2)
    out=head_dir(root,seed,arm,kind); identity=training_identity(root,cfg,arm,kind,seed)
    start,history,best=0,[],None
    if (out/"last.pt").exists():
        saved=read_torch(out/"last.pt",device)
        if saved["identity"]!=identity or saved["best_sha"]!=file_digest(out/"best.pt"):
            raise ValueError("Training resume/checkpoint mismatch")
        model.load_state_dict(saved["state_dict"]); optimizer.load_state_dict(saved["optimizer"]); scheduler.load_state_dict(saved["scheduler"])
        start,history=saved["epoch"],saved["history"]; best=read_torch(out/"best.pt")
        if start>=until:
            snapshot=out/f"through{until}.pt"
            if not snapshot.exists() or read_torch(snapshot)["identity"]!=identity:
                raise ValueError("Requested training snapshot missing")
            return
    for epoch in range(start,until):
        torch.manual_seed(seed+100003*(epoch+1)); model.train()
        loader=torch.utils.data.DataLoader(data,batch_size=cfg["batch_size"],shuffle=True,
            generator=torch.Generator().manual_seed(cfg["data_seed"]+epoch),num_workers=cfg["workers"])
        total,samples=0.,0
        for batch in loader:
            count=len(batch["residual"]); optimizer.zero_grad(set_to_none=True)
            for offset in range(0,count,cfg["micro_batch"]):
                part={k:v[offset:offset+cfg["micro_batch"]].to(device) for k,v in batch.items() if k!="index"}
                x=base.scaled_residual(part,scale,variant)
                loss=(density_loss(model,x,part["coverage"],cfg["tokens_per_view"],cfg["high_weight"])
                      if kind=="density" else text_loss(model,x,part["coverage"],part["typed"],cfg["high_weight"]))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                (loss*len(x)/count).backward(); total+=float(loss.detach())*len(x)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); samples+=count
        calibration=base.fit_calibration(model,kind,cal,scale,variant,cfg,device)
        point=base.validate_scorer(model,kind,val,scale,variant,calibration,cfg,device)
        if best is None or quality(point)>quality(best["point"])+1e-12:
            save_torch(out/"best.pt",dict(identity=identity,state_dict=model.state_dict(),epoch=epoch+1,kind=kind,calibration=calibration,point=point))
            best=read_torch(out/"best.pt")
        history.append(dict(epoch=epoch+1,loss=total/max(1,samples),point=point,selected_epoch=best["epoch"]))
        save_torch(out/"last.pt",dict(identity=identity,state_dict=model.state_dict(),optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),epoch=epoch+1,history=history,best_sha=file_digest(out/"best.pt")))
        atomic_json(out/"history.json",dict(identity=identity,history=history))
        print(f"{arm}/{kind} epoch={epoch+1}/{cfg['density_epochs']} loss={history[-1]['loss']:.5f} selected={best['epoch']} (trained head, no fallback)",flush=True)
    snapshot=out/f"through{until}.pt"
    if snapshot.exists() and file_digest(snapshot)!=file_digest(out/"best.pt"):
        raise ValueError("Immutable snapshot mismatch")
    if not snapshot.exists():
        snapshot.write_bytes((out/"best.pt").read_bytes())


def load_snapshot(root,cfg,arm,kind,seed,until,device):
    path=head_dir(root,seed,arm,kind)/f"through{until}.pt"
    meta=read_torch(path,device)
    if (meta["identity"]!=training_identity(root,cfg,arm,kind,seed) or meta["kind"]!=kind
            or not 1<=meta["epoch"]<=until):
        raise ValueError("Wrong/untrained selected head; distance fallback forbidden")
    model=new_model(root,cfg,kind,seed,device); model.load_state_dict(meta["state_dict"]); model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    return model,meta,file_digest(path)


def decision_path(root,seed,until):
    return Path(root)/"decisions"/f"seed{seed}"/f"through{until}"/"selection.json"


@torch.inference_mode()
def select(args):
    root,cfg=Path(args.work_dir),config(args.work_dir)
    device=base.device_for(args,cfg); until=horizon(args,cfg); arms={}
    for arm,(_,variant) in ARMS.items():
        scale=scale_for(root,cfg,arm,device)
        density,dm,ds=load_snapshot(root,cfg,arm,"density",args.train_seed,until,device)
        text,tm,ts=load_snapshot(root,cfg,arm,"text",args.train_seed,until,device)
        background={k:[] for k in ("visual","full")}
        for item in Features(root,cfg,calibration_rows(root,cfg),arm,False):
            batch=base.batch_to(item,device)
            visual=base.calibrated_map(density,"density",batch,scale,variant,dm["calibration"])
            semantic=base.calibrated_map(text,"text",batch,scale,variant,tm["calibration"])
            for kind,value in (("visual",visual),("full",positive_text_fusion(visual,semantic,cfg["text_strength"]))):
                image=fuse_views(value,batch["boxes"],cfg["eval_size"],variant=="HR",cfg["high_weight"])[0,0].cpu().numpy()
                background[kind].append(image.ravel()[::max(1,image.size//8192)])
        arms[arm]=dict(models=dict(density=ds,text=ts),density_epoch=dm["epoch"],text_epoch=tm["epoch"],density_kind="density",
            density_calibration=dm["calibration"],text_calibration=tm["calibration"],source_density=dm["point"],source_text=tm["point"],
            strength=cfg["text_strength"],thresholds={k:float(np.quantile(np.concatenate(v),1-cfg["val_fpr"],method="higher")) for k,v in background.items()})
    immutable(decision_path(root,args.train_seed,until),dict(fingerprint=digest(cfg),seed=args.train_seed,horizon=until,arms=arms))


def locked_models(root,cfg,seed,until,device):
    decision=load_json(decision_path(root,seed,until))
    if decision["fingerprint"]!=digest(cfg) or decision["seed"]!=seed or decision["horizon"]!=until:
        raise ValueError("Wrong source decision")
    models={}
    for arm in ARMS:
        models[arm]={}
        for kind in ("density","text"):
            model,meta,sha=load_snapshot(root,cfg,arm,kind,seed,until,device)
            if sha!=decision["arms"][arm]["models"][kind]:
                raise ValueError("Selected model changed")
            models[arm][kind]=model
    return decision,models


def prediction_dir(root,seed,until):
    return Path(root)/"predictions"/f"seed{seed}"/f"through{until}"


def prediction_identity(root,cfg,seed,until):
    return dict(fingerprint=digest(cfg),selection=file_digest(decision_path(root,seed,until)),
                eval_seal=file_digest(Path(root)/"sealed_eval.json"),seed=seed,horizon=until)


@torch.inference_mode()
def predict(args):
    root,cfg=Path(args.work_dir),config(args.work_dir); until=horizon(args,cfg)
    device=base.device_for(args,cfg)
    decision,models=locked_models(root,cfg,args.train_seed,until,device)
    identity=prediction_identity(root,cfg,args.train_seed,until)
    out=prediction_dir(root,args.train_seed,until)
    for i,(key,queries) in enumerate(groups(rows(root,"eval")).items()):
        if i%cfg["num_shards"]!=args.shard_id: continue
        pending=[]
        for row in queries:
            path=out/(row["id"]+".npz")
            if path.exists() and path.with_suffix(".json").exists():
                if load_json(path.with_suffix(".json"))!={"identity":identity,"sha":file_digest(path)}:
                    raise ValueError("Prediction changed")
            else:
                if (out/"locked.json").exists(): raise ValueError("Cannot regenerate locked prediction")
                pending.append(row)
        datasets={arm:Features(root,cfg,pending,arm,False) for arm in ARMS}
        scales={arm:scale_for(root,cfg,arm,device) for arm in ARMS}
        for index,row in enumerate(pending):
            maps={}
            for arm,(_,variant) in ARMS.items():
                batch=base.batch_to(datasets[arm][index],device)
                scale=scales[arm]; state=decision["arms"][arm]
                visual=base.calibrated_map(models[arm]["density"],"density",batch,scale,variant,state["density_calibration"])
                semantic=base.calibrated_map(models[arm]["text"],"text",batch,scale,variant,state["text_calibration"])
                for kind,value in (("visual",visual),("full",positive_text_fusion(visual,semantic,state["strength"]))):
                    maps[arm+"_"+kind]=fuse_views(value,batch["boxes"],cfg["eval_size"],variant=="HR",cfg["high_weight"])[0,0].cpu().numpy()
            path=out/(row["id"]+".npz"); atomic_npz(path,**maps)
            immutable(path.with_suffix(".json"),dict(identity=identity,sha=file_digest(path)))
        print(f"predicted {key} seed={args.train_seed}",flush=True)


def lock_predictions(args):
    root,cfg=Path(args.work_dir),config(args.work_dir); until=horizon(args,cfg)
    out=prediction_dir(root,args.train_seed,until); identity=prediction_identity(root,cfg,args.train_seed,until); hashes={}
    for row in rows(root,"eval"):
        path=out/(row["id"]+".npz")
        if load_json(path.with_suffix(".json"))!={"identity":identity,"sha":file_digest(path)}:
            raise ValueError("Invalid prediction before lock")
        with np.load(path,allow_pickle=False) as archive:
            if set(archive.files)!=set(MAPS) or any(archive[n].shape!=(cfg["eval_size"],)*2 or not np.isfinite(archive[n]).all() for n in MAPS):
                raise ValueError("Invalid prediction arrays")
        hashes[row["id"]]=file_digest(path)
    immutable(out/"locked.json",dict(identity=identity,predictions=hashes))


def panels(root,cfg,queries,out,seed,until):
    size=min(256,cfg["eval_size"])
    for row in sorted(queries,key=lambda r:digest([cfg["data_seed"],"factorial_panel",r["id"]]))[:2]:
        with Image.open(row["image_path"]) as image: original=image.convert("RGB").resize((size,size))
        tiles=[("query",original),("GT (evaluation only)",Image.fromarray((evaluation_mask(row,size)*255).astype(np.uint8)).convert("RGB"))]
        with np.load(out/(row["id"]+".npz"),allow_pickle=False) as archive:
            for arm in ARMS:
                value=torch.from_numpy(archive[arm+"_full"])[None,None]
                p=torch.nn.functional.interpolate(value,(size,size),mode="bilinear",align_corners=False)[0,0].sigmoid().numpy()
                color=np.stack([p,.15*(1-p),1-p],-1)*255
                tiles.append((arm+"_full",Image.fromarray((.55*np.asarray(original)+.45*color).clip(0,255).astype(np.uint8))))
        canvas=Image.new("RGB",(3*size,2*(size+24)),"white"); draw=ImageDraw.Draw(canvas)
        for i,(name,tile) in enumerate(tiles):
            x,y=i%3*size,i//3*(size+24); draw.text((x+4,y+4),name,fill="black"); canvas.paste(tile,(x,y+24))
        path=Path(root)/"panels"/f"seed{seed}"/f"through{until}"/(row["dataset"]+"__"+row["category"])/(row["id"]+".png")
        path.parent.mkdir(parents=True,exist_ok=True); canvas.save(path)


def evaluate(args):
    from sklearn.metrics import average_precision_score
    root,cfg=Path(args.work_dir),config(args.work_dir); until=horizon(args,cfg)
    out=prediction_dir(root,args.train_seed,until); lock=load_json(out/"locked.json")
    if lock["identity"]!=prediction_identity(root,cfg,args.train_seed,until): raise ValueError("Stale prediction lock")
    decision=load_json(decision_path(root,args.train_seed,until))
    for i,(key,queries) in enumerate(groups(rows(root,"eval")).items()):
        if i%cfg["num_shards"]!=args.shard_id: continue
        paths=[out/(r["id"]+".npz") for r in queries]
        if any(lock["predictions"][r["id"]]!=file_digest(p) for r,p in zip(queries,paths)):
            raise ValueError("Prediction changed after lock")
        for row in queries: unchanged(row)
        # First target-label content access: all target predictions already locked.
        truth=np.stack([evaluation_mask(r,cfg["eval_size"]) for r in queries]); labels=truth.reshape(len(truth),-1).any(1)
        records=[]
        for name in MAPS:
            values=[]
            for path in paths:
                with np.load(path,allow_pickle=False) as archive: values.append(archive[name].copy())
            maps=np.stack(values); arm,kind=name.split("_"); state=decision["arms"][arm]
            point=base.metric_row(truth,maps,state["thresholds"][kind])
            point.update(I_AP=float(average_precision_score(labels,maps.reshape(len(maps),-1).max(1))) if labels.any() else None,
                arm=arm,score=kind,dataset=key[0],category=key[1],seed=args.train_seed,high_resolution=ARMS[arm][1]=="HR",
                residual=ARMS[arm][0]=="residual",density_kind="density",density_epoch=state["density_epoch"],text_epoch=state["text_epoch"],strength=state["strength"])
            records.append(point)
        immutable(root/"metrics"/f"seed{args.train_seed}"/f"through{until}"/("__".join(key)+".json"),
            dict(prediction_lock_sha=file_digest(out/"locked.json"),metrics=records,
                 evaluation_mask_hashes={r["id"]:file_digest(r["mask_path"]) if r["mask_path"] else None for r in queries}))
        panels(root,cfg,queries,out,args.train_seed,until)
        print(f"evaluated {key} seed={args.train_seed}",flush=True)


def summarize(records,keys):
    grouped=defaultdict(list)
    for record in records: grouped[tuple(record[k] for k in keys)].append(record)
    result=[]
    for key,values in sorted(grouped.items()):
        point=dict(zip(keys,key))
        for metric in METRICS:
            numbers=[r[metric] for r in values if r.get(metric) is not None]
            point[metric]=float(np.mean(numbers)) if numbers else None
            point[metric+"_std"]=float(np.std(numbers)) if numbers else None
        result.append(point)
    return result


def markdown_table(values,category=False):
    lines=["| 数据集"+(" / 类别" if category else "")+" | 评分 | 组别 | 高分辨率 | 残差 | I-AUROC | I-AP | P-AUROC | P-AP | AUPRO | Pixel F1-best* | 背景FPR↓ | 异常像素召回 |",
           "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in values:
        arm=row["arm"]; values=[]
        for metric in METRICS:
            value=row.get(metric); std=row.get(metric+"_std",0)
            values.append("—" if value is None else f"{value*100:.2f}"+(f" ± {100*std:.2f}" if std and not category else ""))
        lines.append("| "+" | ".join([row["dataset"]+(" / "+row["category"] if category else ""),row["score"],arm,
            "✓" if ARMS[arm][1]=="HR" else "—","✓" if ARMS[arm][0]=="residual" else "—"]+values)+" |")
    return lines


def report(args):
    root,cfg=Path(args.work_dir),config(args.work_dir); until=horizon(args,cfg)
    seeds=[int(s) for s in args.train_seeds.split(",")]
    if not seeds or len(set(seeds))!=len(seeds): raise ValueError("Invalid report seeds")
    records=[]
    for seed in seeds:
        lock=prediction_dir(root,seed,until)/"locked.json"
        if load_json(lock)["identity"]!=prediction_identity(root,cfg,seed,until):
            raise ValueError("Stale prediction/selection identity")
        for key in groups(rows(root,"eval")):
            value=load_json(root/"metrics"/f"seed{seed}"/f"through{until}"/("__".join(key)+".json"))
            if value["prediction_lock_sha"]!=file_digest(lock): raise ValueError("Stale evaluation metrics")
            expected={r["id"]:file_digest(r["mask_path"]) if r["mask_path"] else None for r in groups(rows(root,"eval"))[key]}
            if value["evaluation_mask_hashes"]!=expected:
                raise ValueError("Target evaluation masks changed")
            if (len(value["metrics"])!=len(MAPS) or {r["arm"]+"_"+r["score"] for r in value["metrics"]}!=set(MAPS)
                    or any(r["seed"]!=seed or (r["dataset"],r["category"])!=key for r in value["metrics"])):
                raise ValueError("Wrong/missing/duplicate arm metrics")
            records.extend(value["metrics"])
    per_seed=summarize(records,("dataset","score","arm","seed"))
    summary=summarize(per_seed,("dataset","score","arm"))
    category=summarize(records,("dataset","category","score","arm"))
    contrasts=[]
    for dataset in sorted({r["dataset"] for r in summary}):
        for score in ("visual","full"):
            lookup={r["arm"]:r for r in summary if r["dataset"]==dataset and r["score"]==score}
            for name,coeff in (("残差_无HR",{"B":1,"A":-1}),("残差_有HR",{"D":1,"C":-1}),
                ("HR_无残差",{"C":1,"A":-1}),("HR_有残差",{"D":1,"B":-1}),
                ("交互项_D减B减C加A",{"D":1,"B":-1,"C":-1,"A":1})):
                contrasts.append(dict(dataset=dataset,score=score,comparison=name,**{m+"_pp":
                    sum(c*lookup[a][m] for a,c in coeff.items())*100 if all(lookup[a][m] is not None for a in coeff) else None for m in METRICS}))
    out=root/"reports"/f"through{until}"
    for name,values in (("metrics",records),("summary",summary),("per_category",category),("per_seed",per_seed),("contrasts",contrasts)):
        base.write_csv(out/(name+".csv"),values)
    lines=["# 消融试验结果","",f"协议：{PROTOCOL}；工作目录：`{root.resolve()}`；训练至 {until}/{cfg['density_epochs']} 轮；种子：{seeds}。",
        f"文本差分固定，融合权重 λ={cfg['text_strength']}；四组均使用训练后约束器/Flow，不回退最近邻距离。",
        "主表 full 对比完整结构；visual 单独检查视觉分支。G 是仅整图，HR 是整图加四个原图局部裁剪。",
        "数值来自本工作目录已锁定预测的评估文件，不与旧实验拼接。",
        "百分数；类别先宏平均，再跨训练种子平均（±总体标准差）。单种子不代表稳健性。",
        "F1-best* 是使用评估标签得到的最优阈值指标，仅作诊断；FPR/召回使用源正常CAL固定阈值。","",
        "## 数据集汇总","",*markdown_table(sorted(summary,key=lambda r:(r['score']!='full',r['dataset'],r['arm']))),"",
        "## 逐类别结果","",*markdown_table(category,True),"",
        "## 增益对照（百分点）","","| 数据集 | 评分 | 对照 | I-AUROC | P-AP | AUPRO |",
        "|---|---|---|---:|---:|---:|"]
    for row in contrasts:
        lines.append("| "+" | ".join([row['dataset'],row['score'],row['comparison']]+[
            "—" if row[m+"_pp"] is None else f"{row[m+'_pp']:+.2f}" for m in ("I_AUROC","P_AP","PRO_exact")])+" |")
    text="\n".join(lines)+"\n"
    (out/"消融试验.md").write_text(text,encoding="utf-8")
    if args.publish_report:
        publish_markdown(REPO/"消融试验.md",text)
    print(f"Results: {out/'消融试验.md'}",flush=True)


def publish_markdown(template,text):
    template=Path(template); previous=template.read_text(encoding="utf-8")
    start,end="<!-- AUTO_RESULTS_START -->","<!-- AUTO_RESULTS_END -->"
    if previous.count(start)!=1 or previous.count(end)!=1: raise ValueError("Missing/duplicate report markers")
    before,rest=previous.split(start); _,after=rest.split(end)
    temporary=template.with_suffix(".md.tmp")
    temporary.write_text(before+start+"\n\n"+text+"\n"+end+after,encoding="utf-8"); temporary.replace(template)


def package(args):
    root,cfg=Path(args.work_dir),config(args.work_dir); until=horizon(args,cfg)
    path=root/("factorial_analysis_"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")+".zip")
    candidates=[root/n for n in ("config.json","manifest.json","references.json","normal_stats.json","text_bank.json","sealed_source.json","sealed_eval.json")]
    for folder in ("heads","decisions","metrics","reports","panels"):
        candidates.extend(p for p in (root/folder).rglob("*") if p.is_file() and p.suffix in (".json",".csv",".md",".png"))
    candidates.extend((root/"predictions").glob(f"seed*/through{until}/locked.json")); candidates.extend(root.glob("run_*.log"))
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(set(candidates)):
            if item.is_file(): archive.write(item,item.relative_to(root).as_posix())
    print(f"Analysis ZIP (without huge feature/model/prediction tensors): {path}",flush=True)


def parser():
    p=base.parser()
    for action in p._actions:
        if action.dest=="stage":
            action.choices=("prepare","text_bank","export","seal","fit_stats","train","select","predict","lock_predictions","evaluate","report","package")
    p.add_argument("--arm",choices=tuple(ARMS),default="A")
    p.add_argument("--kind",choices=("density","text"),default="density")
    p.add_argument("--text_strength",type=float,default=.5)
    p.add_argument("--publish_report",action="store_true")
    p.set_defaults(work_dir="./checkpoint/factorial_hr_residual_v1",semantic_annotations="")
    return p


if __name__=="__main__":
    arguments=parser().parse_args()
    globals()[arguments.stage](arguments)
