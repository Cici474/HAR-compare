import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# =========================
# 配置
# =========================
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

INERTIAL_FILES = [
    "body_acc_x_", "body_acc_y_", "body_acc_z_",
    "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
    "total_acc_x_", "total_acc_y_", "total_acc_z_",
]


# =========================
# Utils
# =========================
def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def savefig(outdir: Path, name: str):
    path = outdir / name
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    print("[SAVE]", path)

def plot_curves(history: dict, outdir: Path, prefix: str):
    x = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(x, history["train_loss"], marker="o", label="train_loss")
    plt.plot(x, history["val_loss"], marker="o", label="val_loss")
    plt.legend()
    plt.title(f"{prefix} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    savefig(outdir, f"{prefix}_loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(x, history["train_acc"], marker="o", label="train_acc")
    plt.plot(x, history["val_acc"], marker="o", label="val_acc")
    plt.legend()
    plt.title(f"{prefix} Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    savefig(outdir, f"{prefix}_acc_curve.png")
    plt.close()

def plot_confusion(cm, labels, outdir: Path, name: str, title: str):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=7)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.title(title)
    plt.xlabel("Pred")
    plt.ylabel("True")
    savefig(outdir, name)
    plt.close()

def visualize_embeddings(emb, y_1to6, outdir: Path, prefix: str):
    y = np.asarray(y_1to6)

    pca = PCA(n_components=2, random_state=SEED)
    z_pca = pca.fit_transform(emb)

    plt.figure()
    for lab in sorted(np.unique(y)):
        idx = y == lab
        plt.scatter(z_pca[idx, 0], z_pca[idx, 1], s=6, label=f"{lab}:{ACTIVITY_NAMES[int(lab)]}")
    plt.legend(markerscale=3, fontsize=8)
    plt.title(f"{prefix} Embedding PCA(2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    savefig(outdir, f"{prefix}_emb_pca2.png")
    plt.close()

    tsne = TSNE(
        n_components=2,
        perplexity=35,
        learning_rate="auto",
        init="pca",
        random_state=SEED
    )
    z_tsne = tsne.fit_transform(emb)

    plt.figure()
    for lab in sorted(np.unique(y)):
        idx = y == lab
        plt.scatter(z_tsne[idx, 0], z_tsne[idx, 1], s=6, label=f"{lab}:{ACTIVITY_NAMES[int(lab)]}")
    plt.legend(markerscale=3, fontsize=8)
    plt.title(f"{prefix} Embedding t-SNE(2D)")
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    savefig(outdir, f"{prefix}_emb_tsne2.png")
    plt.close()


# =========================
# 定位路径
# =========================
def find_ucihar_root(start_path: str) -> Path:
    p = Path(start_path)
    if p.is_file():
        p = p.parent
    for cur in [p] + list(p.parents):
        if (cur / "features.txt").exists() and \
           (cur / "train" / "X_train.txt").exists() and \
           (cur / "test" / "X_test.txt").exists():
            return cur
    raise FileNotFoundError("找不到 UCI HAR 根目录（需要 features.txt + train/test 下的 X/y/subject 文件）")


# =========================
# 加载数据
# =========================
def load_features_names(root: Path):
    feats = pd.read_csv(root / "features.txt", sep=r"\s+", header=None, names=["idx", "name"])
    name_counts = {}
    fixed = []
    for n in feats["name"].tolist():
        if n not in name_counts:
            name_counts[n] = 0
            fixed.append(n)
        else:
            name_counts[n] += 1
            fixed.append(f"{n}__dup{name_counts[n]}")
    return fixed

def load_feature_dataset(root: Path):
    names = load_features_names(root)
    X_train = np.loadtxt(root / "train" / "X_train.txt")
    y_train = np.loadtxt(root / "train" / "y_train.txt").astype(int)
    X_test  = np.loadtxt(root / "test" / "X_test.txt")
    y_test  = np.loadtxt(root / "test" / "y_test.txt").astype(int)
    subj_train = np.loadtxt(root / "train" / "subject_train.txt").astype(int)
    subj_test  = np.loadtxt(root / "test" / "subject_test.txt").astype(int)
    return X_train, y_train, subj_train, X_test, y_test, subj_test, names

def load_inertial_split(root: Path, split: str):
    base = root / split / "Inertial Signals"
    mats = []
    for prefix in INERTIAL_FILES:
        p = base / f"{prefix}{split}.txt"
        mats.append(np.loadtxt(p))  # (N,128)
    X = np.stack(mats, axis=0).transpose(1, 0, 2)  # (N,9,128)
    y = np.loadtxt(root / split / f"y_{split}.txt").astype(int)
    subj = np.loadtxt(root / split / f"subject_{split}.txt").astype(int)
    return X, y, subj

def load_inertial_dataset(root: Path):
    X_train, y_train, s_train = load_inertial_split(root, "train")
    X_test, y_test, s_test = load_inertial_split(root, "test")
    return X_train, y_train, s_train, X_test, y_test, s_test


# =========================
# 数据预处理
# =========================
class NumpyDataset(Dataset):
    def __init__(self, X, y_1to6):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_1to6 - 1, dtype=torch.long)  # 0..5
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

class InertialDataset(Dataset):
    def __init__(self, X, y_1to6, augment=False, jitter_std=0.02, scale_std=0.08):
        self.X = X.astype(np.float32)
        self.y = (y_1to6 - 1).astype(np.int64)
        self.augment = augment
        self.jitter_std = jitter_std
        self.scale_std = scale_std

    def __len__(self): return self.X.shape[0]

    def _aug(self, x):
        # jitter: 加小高斯噪声
        if self.jitter_std > 0:
            x = x + np.random.randn(*x.shape).astype(np.float32) * self.jitter_std
        # scale: 每个通道随机缩放
        if self.scale_std > 0:
            scale = (1.0 + np.random.randn(x.shape[0], 1).astype(np.float32) * self.scale_std)
            x = x * scale
        return x

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment:
            x = self._aug(x)
        return torch.tensor(x), torch.tensor(self.y[i])

# =========================
# 模型加载
# =========================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=561, num_classes=6):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        )
        self.head = nn.Linear(128, num_classes)
    def forward(self, x, return_emb=False):
        emb = self.backbone(x)
        logits = self.head(emb)
        return (logits, emb) if return_emb else logits

class CNN1DClassifier(nn.Module):
    def __init__(self, in_ch=9, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_emb = nn.Linear(128, 128)
        self.head = nn.Linear(128, num_classes)
    def forward(self, x, return_emb=False):
        h = self.features(x).squeeze(-1)
        emb = F.relu(self.fc_emb(h))
        logits = self.head(emb)
        return (logits, emb) if return_emb else logits

class Chomp1d(nn.Module):#截断填充，保证输出长度与输入一致
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):#残差块，解决深层网络的梯度消失问题，同时融合浅层特征
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.drop(F.relu(self.bn1(self.chomp1(self.conv1(x)))))
        out = self.drop(F.relu(self.bn2(self.chomp2(self.conv2(out)))))
        res = x if self.down is None else self.down(x)
        return F.relu(out + res)

class TCNClassifier(nn.Module):
    def __init__(self, in_ch=9, num_classes=6, channels=(64, 128, 128), kernel_size=5, dropout=0.2):
        super().__init__()
        blocks = []
        prev = in_ch
        for i, ch in enumerate(channels):
            blocks.append(TemporalBlock(prev, ch, kernel_size, dilation=2**i, dropout=dropout)) #快速扩大感受野
            prev = ch
        self.tcn = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_emb = nn.Linear(prev, 128)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x, return_emb=False):
        h = self.tcn(x)
        h = self.pool(h).squeeze(-1)
        emb = F.relu(self.fc_emb(h))
        logits = self.head(emb)
        return (logits, emb) if return_emb else logits


# =========================
# 训练
# =========================
def acc_from_logits(logits, y):
    return (torch.argmax(logits, dim=1) == y).float().mean().item()

@torch.no_grad()
def eval_model(model, loader, device, criterion):
    model.eval()
    losses, accs = [], []
    ys, ps = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        losses.append(criterion(logits, y).item())
        accs.append(acc_from_logits(logits, y))
        ys.append(y.cpu().numpy())
        ps.append(torch.argmax(logits, dim=1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return float(np.mean(losses)), float(np.mean(accs)), y_true, y_pred

def class_weights(y0to5):
    c = np.bincount(y0to5, minlength=6).astype(np.float32)
    w = c.sum() / (c + 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

def train_loop(
    model, train_loader, val_loader, device,
    epochs=40, lr=1e-3, weight_decay=1e-4, patience=7, weights=None
):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, verbose=True)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, bad = -1.0, 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tl, ta = [], []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(loss.item())
            ta.append(acc_from_logits(logits, y))

        vl, va, _, _ = eval_model(model, val_loader, device, criterion)
        sch.step(va)

        hist["train_loss"].append(float(np.mean(tl)))
        hist["train_acc"].append(float(np.mean(ta)))
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)

        print(f"[Epoch {ep:02d}] train_loss={hist['train_loss'][-1]:.4f} train_acc={hist['train_acc'][-1]:.4f} | "
              f"val_loss={vl:.4f} val_acc={va:.4f}")

        if va > best + 1e-4:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[EarlyStopping] stop at epoch {ep}, best_val_acc={best:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist

@torch.no_grad()
def extract_emb(model, loader, device):
    model.eval()
    embs, ys = [], []
    for X, y in loader:
        X = X.to(device)
        logits, emb = model(X, return_emb=True)
        embs.append(emb.cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.vstack(embs), np.concatenate(ys)

# =========================
# 评估
# =========================
def subject_wise_split(X, y, subjects, test_size=0.2):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr_idx, val_idx = next(gss.split(X, y, groups=subjects))
    return tr_idx, val_idx

def run_mlp(root: Path, outdir: Path, device: str):
    X_train, y_train, subj_train, X_test, y_test, subj_test, names = load_feature_dataset(root)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    tr_idx, val_idx = subject_wise_split(X_train, y_train, subj_train)
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    train_loader = DataLoader(NumpyDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    val_loader   = DataLoader(NumpyDataset(X_val, y_val), batch_size=512, shuffle=False)
    test_loader  = DataLoader(NumpyDataset(X_test, y_test), batch_size=512, shuffle=False)

    w = class_weights((y_tr - 1).astype(np.int64))
    model = MLPClassifier(in_dim=X_tr.shape[1], num_classes=6)
    model, hist = train_loop(model, train_loader, val_loader, device, epochs=40, lr=1e-3, patience=7, weights=w)

    plot_curves(hist, outdir, "mlp")

    crit = nn.CrossEntropyLoss(weight=w.to(device))
    tl, ta, y0, p0 = eval_model(model, test_loader, device, crit)
    labels = [ACTIVITY_NAMES[i] for i in range(1, 7)]
    cm = confusion_matrix(y0, p0)
    plot_confusion(cm, labels, outdir, "mlp_confusion.png", f"MLP Confusion (acc={ta:.3f})")

    report = classification_report(y0, p0, target_names=labels, digits=4)
    (outdir / "mlp_report.txt").write_text(report, encoding="utf-8")

    emb, y_emb = extract_emb(model, test_loader, device)
    visualize_embeddings(emb, y_emb + 1, outdir, "mlp_test")

    macro = f1_score(y0, p0, average="macro")
    return {"model": "MLP(561)", "test_acc": ta, "macro_f1": macro}

def norm_inertial(trainX, testX):
    mean = trainX.mean(axis=(0, 2), keepdims=True)
    std = trainX.std(axis=(0, 2), keepdims=True) + 1e-8
    return (trainX - mean) / std, (testX - mean) / std

def run_cnn_or_tcn(root: Path, outdir: Path, device: str, kind: str):
    X_train, y_train, s_train, X_test, y_test, s_test = load_inertial_dataset(root)
    X_train, X_test = norm_inertial(X_train, X_test)

    tr_idx, val_idx = subject_wise_split(X_train, y_train, s_train)
    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    train_ds = InertialDataset(X_tr, y_tr, augment=True, jitter_std=0.02, scale_std=0.08)
    val_ds   = InertialDataset(X_val, y_val, augment=False)
    test_ds  = InertialDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

    w = class_weights((y_tr - 1).astype(np.int64))

    if kind == "CNN1D":
        model = CNN1DClassifier(in_ch=9, num_classes=6)
        lr = 8e-4
    elif kind == "TCN":
        model = TCNClassifier(in_ch=9, num_classes=6, channels=(64,128,128), kernel_size=5, dropout=0.2)
        lr = 8e-4
    else:
        raise ValueError("kind must be CNN1D or TCN")

    model, hist = train_loop(model, train_loader, val_loader, device, epochs=50, lr=lr, patience=7, weights=w)

    plot_curves(hist, outdir, kind.lower())

    crit = nn.CrossEntropyLoss(weight=w.to(device))
    tl, ta, y0, p0 = eval_model(model, test_loader, device, crit)

    labels = [ACTIVITY_NAMES[i] for i in range(1, 7)]
    cm = confusion_matrix(y0, p0)
    plot_confusion(cm, labels, outdir, f"{kind.lower()}_confusion.png", f"{kind} Confusion (acc={ta:.3f})")

    report = classification_report(y0, p0, target_names=labels, digits=4)
    (outdir / f"{kind.lower()}_report.txt").write_text(report, encoding="utf-8")

    emb, y_emb = extract_emb(model, test_loader, device)
    visualize_embeddings(emb, y_emb + 1, outdir, f"{kind.lower()}_test")

    torch.save(model.state_dict(), outdir / f"{kind.lower()}_best.pt")

    macro = f1_score(y0, p0, average="macro")
    return {"model": kind, "test_acc": ta, "macro_f1": macro}

# =========================
# 主流程
# =========================
def main():
    START_PATH = r"E:\yan\class\data data\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\Inertial Signals"
    root = find_ucihar_root(START_PATH)
    print("[INFO] root =", root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device =", device)

    base_out = ensure_outdir(Path("./outputs_highscore"))

    results = []

    # 1) MLP
    out_mlp = ensure_outdir(base_out / "MLP")
    print("\n===== Run MLP(561) =====")
    results.append(run_mlp(root, out_mlp, device))

    # 2) CNN1D
    out_cnn = ensure_outdir(base_out / "CNN1D")
    print("\n===== Run CNN1D =====")
    results.append(run_cnn_or_tcn(root, out_cnn, device, "CNN1D"))

    # 3) TCN
    out_tcn = ensure_outdir(base_out / "TCN")
    print("\n===== Run TCN =====")
    results.append(run_cnn_or_tcn(root, out_tcn, device, "TCN"))

    df = pd.DataFrame(results).sort_values(by=["test_acc", "macro_f1"], ascending=False)
    df.to_csv(base_out / "summary.csv", index=False)
    print("\n[SAVE]", base_out / "summary.csv")
    print(df)

    best = df.iloc[0].to_dict()
    print("\n[BEST MODEL]", best)

if __name__ == "__main__":
    main()
