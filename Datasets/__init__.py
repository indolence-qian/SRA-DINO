import torch

from Datasets import (
    mvtec, visa, btad, mpdd, 
    tn3k, clinicdb, colondb, isic
)

# (Dataset, Split, root_path) 
root_dir = "/Data"
DATASET_REGISTRY = {
    "mvtec":       (mvtec.Dataset, mvtec.DatasetSplit, f"/home/kexin/hd1/Tate_qfg/data/mvtec"),
    "visa":        (visa.Dataset, visa.DatasetSplit, f"/home/kexin/hd1/Tate_qfg/data/VisA"),
    "btad":        (btad.Dataset, btad.DatasetSplit, f"{root_dir}/Industrial_Dataset/BTAD/BTech_Dataset_transformed"),
    "mpdd":        (mpdd.Dataset, mpdd.DatasetSplit, f"{root_dir}/Industrial_Dataset/MPDD"),
    "tn3k":        (tn3k.Dataset, tn3k.DatasetSplit, f"{root_dir}/Medical_Dataset/TN3K"),
    "clinicdb":    (clinicdb.Dataset, clinicdb.DatasetSplit, f"{root_dir}/Medical_Dataset/CVC-ClinicDB"),
    "colondb":     (colondb.Dataset, colondb.DatasetSplit, f"{root_dir}/Medical_Dataset/CVC-ColonDB"),
    "isic":        (isic.Dataset, isic.DatasetSplit, f"{root_dir}/Medical_Dataset/ISIC"),
}

DATASET_CLASSES = {
    "mvtec":    {'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper'},
    "visa":     {"candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2", "pcb1","pcb2", "pcb3", "pcb4", "pipe_fryum"},
    "btad":     {'01', '02', '03'},
    "mpdd":     {'bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes'},
    "tn3k":     {'01'},
    "clinicdb": {'01'},
    "colondb":  {'01'},
    "isic":     {'01'},
}