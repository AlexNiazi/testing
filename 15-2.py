# train_brain_tumor_pytorch_v2b3.py
# PyTorch port for runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
# - EfficientNetV2-B3 (timm), 300x300
# - Stratified split, balanced sampling, MixUp, AMP
# - Adam optimizer (head then full FT)
# - TTA at test, confusion matrix & per-class report

import os, re, random, itertools, warnings, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Optional DICOM
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    HAS_DICOM = True
except Exception:
    HAS_DICOM = False
    warnings.warn("pydicom not installed; DICOM images won't be decoded.", RuntimeWarning)

# -------------------- Config --------------------
SEED = 101
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True

LABELS = [
    'Normal', 'Astrocitoma', 'Carcinoma', 'Ependimoma',
    'Ganglioglioma', 'Germinoma', 'Glioblastoma', 'Granuloma',
    'Meduloblastoma', 'Meningioma', 'Neurocitoma', 'Oligodendroglioma',
    'Papiloma', 'Schwannoma', 'Tuberculoma'
]
NUM_CLASSES = len(LABELS)
CLS2ID = {c:i for i,c in enumerate(LABELS)}

# >>>>>>>>>>>> CHANGE THIS <<<<<<<<<<<<<
DATA_DIR = "/workspace/1/15 data copy"  # patched
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

VAL_SPLIT = 0.15
TEST_SPLIT = 0.10

IMG_SIZE = 300
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))
EPOCHS_HEAD = 40
EPOCHS_FT   = 12
MODEL_OUT   = "best_model.pth"
USE_25D = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# -------------------- File listing --------------------
def list_files_and_labels(root_dir, labels):
    files, y = [], []
    for cls in labels:
        cdir = os.path.join(root_dir, cls)
        if not os.path.isdir(cdir):
            print(f"[WARN] missing class folder: {cdir}")
            continue
        for name in os.listdir(cdir):
            if name.startswith("."): continue
            p = os.path.join(cdir, name)
            if not os.path.isfile(p): continue
            pl = name.lower()
            if pl.endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff",".gif",".webp",".dcm",".dicom")):
                files.append(p); y.append(CLS2ID[cls])
    return np.array(files), np.array(y, dtype=np.int64)

# -------------------- DICOM / 2.5D helpers --------------------
_num_suffix_re = re.compile(r"^(?P<stem>.*?)(?P<num>\d+)(?P<ext>\.[^.]+)$")

def _neighbors_from_filename(p: str):
    b = os.path.basename(p)
    m = _num_suffix_re.match(b)
    if not m: return None
    stem, num_s, ext = m.group("stem"), m.group("num"), m.group("ext")
    num = int(num_s); pdir = os.path.dirname(p)
    p_prev = os.path.join(pdir, f"{stem}{num-1:0{len(num_s)}d}{ext}")
    p_next = os.path.join(pdir, f"{stem}{num+1:0{len(num_s)}d}{ext}")
    return p_prev, p, p_next

def _dicom_to_uint8_rgb(path: str) -> np.ndarray:
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    inter = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + inter
    if apply_voi_lut is not None:
        try: arr = apply_voi_lut(arr, ds).astype(np.float32)
        except Exception: pass
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)
    arr = (arr * 255.0).astype(np.uint8)
    return np.stack([arr, arr, arr], axis=-1)

def _read_one_image_uint8_rgb(path: str) -> np.ndarray:
    pl = path.lower()
    if HAS_DICOM and (pl.endswith(".dcm") or pl.endswith(".dicom")):
        return _dicom_to_uint8_rgb(path)
    # Try PIL first for better EXIF handling
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            return np.array(img, dtype=np.uint8)
    except Exception:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Corrupt image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def _read_25d_uint8_rgb(path: str) -> np.ndarray:
    nbrs = _neighbors_from_filename(path)
    if not nbrs:
        return _read_one_image_uint8_rgb(path)
    p_prev, p_cur, p_next = nbrs
    def gray_or_none(pth):
        try:
            im = _read_one_image_uint8_rgb(pth)
            return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        except Exception:
            return None
    g_cur = gray_or_none(p_cur)
    g_prev = gray_or_none(p_prev) or g_cur
    g_next = gray_or_none(p_next) or g_cur
    return np.stack([g_prev, g_cur, g_next], axis=-1).astype(np.uint8)

# -------------------- Dataset --------------------
class BrainTumorDS(Dataset):
    def __init__(self, paths, labels, training: bool):
        self.paths = list(paths); self.labels = list(labels)
        self.training = training

        # Albumentations: keep dtype-safe ops
        self.train_tf = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.CoarseDropout(max_holes=4, max_height=IMG_SIZE//10, max_width=IMG_SIZE//10, fill_value=0, p=0.15),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
        self.val_tf = A.Compose([
            A.LongestMaxSize(max_size=IMG_SIZE),
            A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]; y = self.labels[idx]
        try:
            img = _read_25d_uint8_rgb(p) if USE_25D else _read_one_image_uint8_rgb(p)
        except Exception as e:
            # fallback black
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
        tfm = self.train_tf if self.training else self.val_tf
        out = tfm(image=img)
        x = out["image"]  # CxHxW float32
        return x, y

# -------------------- MixUp --------------------
def mixup_batch(x, y, alpha=0.2, num_classes=NUM_CLASSES):
    if alpha <= 0: return x, F.one_hot(y, num_classes=num_classes).float(), None
    lam = np.random.beta(alpha, alpha)
    bs = x.size(0)
    index = torch.randperm(bs, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_one = F.one_hot(y, num_classes=num_classes).float()
    y_shf = y_one[index, :]
    mixed_y = lam * y_one + (1.0 - lam) * y_shf
    return mixed_x, mixed_y, lam

# -------------------- Utilities --------------------
def set_requires_grad(model, flag):
    for p in model.parameters():
        p.requires_grad = flag

def create_model(num_classes):
    # EfficientNetV2-B3 pretrained
    model = timm.create_model("tf_efficientnetv2_b3", pretrained=True, in_chans=3, num_classes=num_classes)
    return model

def get_class_weights(labels):
    # Balanced sampling: inverse frequency per class
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    counts = np.where(counts==0, 1, counts)
    weights_per_class = 1.0 / counts
    sample_weights = weights_per_class[labels]
    return torch.DoubleTensor(sample_weights)

# -------------------- Main --------------------
def main():
    # 1) Files
    filepaths, targets = list_files_and_labels(DATA_DIR, LABELS)
    print("Total images:", len(filepaths))
    assert len(filepaths) > 0, "No images found. Check DATA_DIR and class folder names."

    # 2) Stratified split
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        filepaths, targets, test_size=(VAL_SPLIT + TEST_SPLIT),
        random_state=SEED, stratify=targets
    )
    rel = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT + 1e-9)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=(1.0 - rel),
        random_state=SEED, stratify=y_tmp
    )
    print(f"Stratified image split -> Train/Val/Test: {len(X_train)} {len(X_val)} {len(X_test)}")

    # 3) Datasets / Loaders
    train_ds = BrainTumorDS(X_train, y_train, training=True)
    val_ds   = BrainTumorDS(X_val,   y_val,   training=False)
    test_ds  = BrainTumorDS(X_test,  y_test,  training=False)

    # Balanced sampler
    sample_weights = get_class_weights(y_train)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # 4) Model
    model = create_model(NUM_CLASSES).to(DEVICE)
    print("Params:", sum(p.numel() for p in model.parameters())/1e6, "M")

    # Split head/backbone (timm head usually 'classifier' or 'fc')
    head_names = ["classifier", "fc", "head", "final_layer"]
    head_params = []
    backbone_params = []
    for n, p in model.named_parameters():
        if any(hn in n for hn in head_names):
            head_params.append(p)
        else:
            backbone_params.append(p)

    # 5) Loss & metrics
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()

    # 6) Phase 1: train head (freeze backbone)
    set_requires_grad(model, False)
    for p in head_params: p.requires_grad = True
    opt = torch.optim.Adam(head_params, lr=1e-3)

    best_val_acc = 0.0
    torch.cuda.empty_cache()

    def run_epoch(loader, train_mode=True, use_mixup=True):
        if train_mode:
            model.train()
        else:
            model.eval()
        total, correct, loss_sum = 0, 0, 0.0

        for (x, y) in tqdm(loader, disable=False):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            if train_mode and use_mixup:
                xm, ym, _ = mixup_batch(x, y, alpha=0.2, num_classes=NUM_CLASSES)
                target = ym
            else:
                xm, target = x, F.one_hot(y, num_classes=NUM_CLASSES).float()

            with autocast(enabled=True):
                logits = model(xm)
                loss = -(F.log_softmax(logits, dim=1) * target).sum(dim=1).mean()  # soft target CE

            if train_mode:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            if train_mode and use_mixup:
                # approximate accuracy: compare to hard labels (not perfect under mixup)
                correct += (preds == y).sum().item()
            else:
                correct += (preds == y).sum().item()
            total += x.size(0)

        return loss_sum/total, correct/total

    print("\n== Stage 1: head-only training ==")
    for epoch in range(1, EPOCHS_HEAD+1):
        tr_loss, tr_acc = run_epoch(train_loader, train_mode=True, use_mixup=True)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(val_loader, train_mode=False, use_mixup=False)
        print(f"[Head {epoch:03d}] train_loss {tr_loss:.4f} acc {tr_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print("  -> saved:", MODEL_OUT)

    # 7) Phase 2: full fine-tune
    print("\n== Stage 2: full fine-tune ==")
    set_requires_grad(model, True)
    # cosine LR with restarts
    steps_per_epoch = max(1, math.ceil(len(train_loader)))
    t0 = steps_per_epoch * 3
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer := torch.optim.Adam(model.parameters(), lr=1e-5),
        T_0=t0, T_mult=1
    )

    for epoch in range(1, EPOCHS_FT+1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for i, (x, y) in enumerate(tqdm(train_loader)):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            xm, ym, _ = mixup_batch(x, y, alpha=0.2, num_classes=NUM_CLASSES)
            with autocast(True):
                logits = model(xm)
                loss = -(F.log_softmax(logits, dim=1) * ym).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_sched.step(epoch - 1 + i/steps_per_epoch)
            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        with torch.no_grad():
            val_loss, val_acc = 0.0, 0.0
            vv_total, vv_correct = 0, 0
            model.eval()
            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
                with autocast(True):
                    logits = model(x)
                    loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                vv_total += x.size(0)
                vv_correct += (logits.argmax(1) == y).sum().item()
            val_loss /= vv_total; val_acc = vv_correct / vv_total

        print(f"[FT {epoch:03d}] train_loss {loss_sum/total:.4f} acc {correct/total:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_OUT)
            print("  -> saved:", MODEL_OUT)

    # 8) Evaluate on test (with TTA)
    print("\nEvaluating best model on test set…")
    model.load_state_dict(torch.load(MODEL_OUT, map_location=DEVICE))
    model.eval()

    # simple TTA: 8 random light augs at test-time (reuse train tf but lighter)
    tta_tf = A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE),
        A.PadIfNeeded(IMG_SIZE, IMG_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=5, p=0.5),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

    def tta_predict(x_batch, n=8):
        # x_batch: already tensor images (normalized) -> for TTA we need original imgs
        # Here, for simplicity, we re-run DS transforms via OpenCV/PIL would be complex in loader
        # Instead: do standard (no extra random ops) TTA by flips
        logits_list = []
        with torch.no_grad(), autocast(True):
            # base
            logits_list.append(model(x_batch))
            # hflip
            logits_list.append(model(torch.flip(x_batch, dims=[3])))
            # vflip
            logits_list.append(model(torch.flip(x_batch, dims=[2])))
        return torch.stack(logits_list, 0).mean(0)

    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.numpy()
            logits = tta_predict(x, n=3)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pred = probs.argmax(1)
            y_true_all.append(y); y_pred_all.append(pred)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    # Reports
    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(NUM_CLASSES)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    print("\nClassification report:")
    rep = classification_report(y_true_all, y_pred_all, target_names=LABELS, digits=4)
    print(rep)
    rep_dict = classification_report(y_true_all, y_pred_all, target_names=LABELS, output_dict=True, digits=4)
    pd.DataFrame(rep_dict).transpose().to_csv("per_class_report.csv", index=True)
    print("Saved per-class metrics to per_class_report.csv")

    # Confusion matrix plot
    def plot_confusion_matrix_percent(cm_row_norm, classes, title='Confusion Matrix (%)', cmap=plt.cm.Blues, figsize=(11,9)):
        cm_pct = cm_row_norm * 100.0
        plt.figure(figsize=figsize)
        plt.imshow(cm_pct, interpolation='nearest', cmap=cmap)
        plt.title(title)
        cb = plt.colorbar(fraction=0.046, pad=0.04)
        cb.set_label('%', rotation=0, labelpad=10)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        thresh = (cm_pct.max() if cm_pct.size else 0) / 2.0
        for i, j in itertools.product(range(cm_pct.shape[0]), range(cm_pct.shape[1])):
            val = cm_pct[i, j]
            plt.text(j, i, f"{val:.1f}%", ha="center",
                     color="white" if val > thresh else "black", fontsize=9)
        plt.ylabel('True label'); plt.xlabel('Predicted label')
        plt.tight_layout(); plt.savefig("confusion_matrix_percent.png", dpi=150)

    plot_confusion_matrix_percent(cm_norm, LABELS)
    plt.show()
    print("Saved: confusion_matrix_percent.png")

if __name__ == "__main__":
    main()
