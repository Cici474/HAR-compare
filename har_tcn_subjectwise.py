import os
from pathlib import Path
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


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

def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

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

def visualize_embeddings(emb, y_1to6, outdir: Path, prefix: str, random_state=0):
    y = np.asarray(y_1to6)

    # PCA 2D
    pca = PCA(n_components=2, random_state=random_state)
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

    # t-SNE 2D
    tsne = TSNE(
        n_components=2,
        perplexity=35,
        learning_rate="auto",
        init="pca",
        random_state=random_state
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
# 自动定位 UCI HAR 根目录
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

    raise FileNotFoundError(
        "找不到 UCI HAR 根目录。请确保 start_path 在数据集目录内部。\n"
        "根目录应包含 features.txt、train/X_train.txt、test/X_test.txt"
    )

# =========================
# 读取 inertial signals + y + subject，惯性信号分布在9个文件中，批量读取并拼接
# =========================
def load_inertial_split(root: Path, split: str):
    base = root / split / "Inertial Signals"
    if not base.exists():
        raise FileNotFoundError(f"找不到 Inertial Signals：{base}")

    mats = []
    for prefix in INERTIAL_FILES:
        p = base / f"{prefix}{split}.txt"
        if not p.exists():
            raise FileNotFoundError(f"缺少文件：{p}")
        m = np.loadtxt(p)  # (N, 128)
        mats.append(m)

    X = np.stack(mats, axis=0).transpose(1, 0, 2)  # (N, 9, 128)
    y = np.loadtxt(root / split / f"y_{split}.txt").astype(int)
    subj = np.loadtxt(root / split / f"subject_{split}.txt").astype(int)
    return X, y, subj

def load_inertial_dataset(root: Path):
    X_train, y_train, s_train = load_inertial_split(root, "train")
    X_test, y_test, s_test = load_inertial_split(root, "test")
    return X_train, y_train, s_train, X_test, y_test, s_test

# =========================
# Dataset + 轻量增强
# =========================
class InertialDataset(Dataset):
    def __init__(self, X, y_1to6, augment=False, jitter_std=0.02, scale_std=0.10):
        self.X = X.astype(np.float32)      # (N, 9, 128)
        self.y = (y_1to6 - 1).astype(np.int64)  # 0..5
        self.augment = augment
        self.jitter_std = jitter_std
        self.scale_std = scale_std

    def __len__(self):
        return self.X.shape[0]
    def _augment(self, x):
        # x: (9, 128)
        # jitter: 加小高斯噪声
        if self.jitter_std > 0:
           x = x + np.random.randn(*x.shape).astype(np.float32) * self.jitter_std
        # scale :
        if self.scale_std > 0:
        scale = (1.0 + np.random.randn(x.shape[0], 1).astype(np.float32) * self.scale_std)
        x = x * scale
        return x

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = self._augment(x)
        return torch.tensor(x), torch.tensor(self.y[idx])

# =========================
# 模型：TCN Residual Block
# =========================
class Chomp1d(nn.Module): #截断填充，保证输出长度与输入一致
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module): #残差块，解决深层网络的梯度消失问题，同时融合浅层特征
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)

        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)

class TCNClassifier(nn.Module):
    def __init__(self, in_ch=9, num_classes=6, channels=(64, 128, 128), kernel_size=5, dropout=0.2):
        super().__init__()
        blocks = []
        prev = in_ch
        for i, ch in enumerate(channels):
            dilation = 2 ** i  #快速扩大感受野
            blocks.append(TemporalBlock(prev, ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            prev = ch
        self.tcn = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1) #自适应平均池化
        self.fc_emb = nn.Linear(prev, 128)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x, return_emb=False):
        # x: (B, 9, 128)
        h = self.tcn(x)              # (B, C, 128)
        h = self.pool(h).squeeze(-1) # (B, C)
        emb = F.relu(self.fc_emb(h)) # (B, 128)
        logits = self.head(emb)
        if return_emb:
            return logits, emb
        return logits

# =========================
# 训练/评估
# =========================
def accuracy_from_logits(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()

@torch.no_grad()
def eval_model(model, loader, device, criterion):
    model.eval()
    losses, accs = [], []
    all_pred, all_true = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y).item()
        acc = accuracy_from_logits(logits, y)
        losses.append(loss)
        accs.append(acc)
        all_pred.append(torch.argmax(logits, dim=1).cpu().numpy())
        all_true.append(y.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    return float(np.mean(losses)), float(np.mean(accs)), all_true, all_pred

def compute_class_weights(y0to5):
    # 反频率权重，提升难类（例如 sitting/standing）被关注程度
    counts = np.bincount(y0to5, minlength=6).astype(np.float32)
    w = counts.sum() / (counts + 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    class_weights,
    epochs=40,
    lr=8e-4,
    weight_decay=1e-4,
    patience=7,
):
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    best_state = None
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        model.train()
        train_losses, train_accs = [], []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 稳定训练
            optimizer.step()

            train_losses.append(loss.item())
            train_accs.append(accuracy_from_logits(logits, y))

        val_loss, val_acc, _, _ = eval_model(model, val_loader, device, criterion)
        scheduler.step(val_acc)

        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(float(np.mean(train_losses)))
        history["train_acc"].append(float(np.mean(train_accs)))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)

        print(f"[Epoch {ep:02d}] "
              f"train_loss={history['train_loss'][-1]:.4f} train_acc={history['train_acc'][-1]:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={cur_lr:.2e}")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[EarlyStopping] Stop at epoch={ep}, best_val_acc={best_val_acc:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs, ys = [], []
    for X, y in loader:
        X = X.to(device)
        logits, emb = model(X, return_emb=True)
        embs.append(emb.cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.vstack(embs), np.concatenate(ys)


# =========================
# 主流程：Subject-wise split + TCN + 输出
# =========================
def main():
    START_PATH = r"E:\yan\class\data data\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\Inertial Signals"

    outdir = ensure_outdir(Path("./outputs_tcn"))
    root = find_ucihar_root(START_PATH)
    print("[INFO] UCI HAR root =", root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device =", device)

    # 读取 train/test（train 用来做 subject-wise train/val，test 保持官方 test）
    X_train, y_train, s_train, X_test, y_test, s_test = load_inertial_dataset(root)
    print("[INFO] train:", X_train.shape, "test:", X_test.shape)

    # 归一化（按通道统计 train mean/std）
    ch_mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, 9, 1)
    ch_std  = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train = (X_train - ch_mean) / ch_std
    X_test  = (X_test  - ch_mean) / ch_std

    # Subject-wise split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    (tr_idx, val_idx) = next(gss.split(X_train, y_train, groups=s_train))

    X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    # 类权重（提升坐/站等难类关注度）
    class_w = compute_class_weights((y_tr - 1).astype(np.int64))
    print("[INFO] class_weights =", class_w.numpy().round(3).tolist())

    # 数据加载
    train_ds = InertialDataset(X_tr, y_tr, augment=True, jitter_std=0.02, scale_std=0.08)
    val_ds   = InertialDataset(X_val, y_val, augment=False)
    test_ds  = InertialDataset(X_test, y_test, augment=False)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

    # TCN
    model = TCNClassifier(in_ch=9, num_classes=6, channels=(64, 128, 128), kernel_size=5, dropout=0.2)

    model, history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        class_weights=class_w,
        epochs=50,
        lr=8e-4,
        weight_decay=1e-4,
        patience=7,
    )

    # 保存曲线
    plot_curves(history, outdir, prefix="tcn")

    # 评估
    criterion_test = nn.CrossEntropyLoss(weight=class_w.to(device))
    test_loss, test_acc, y_true0, y_pred0 = eval_model(model, test_loader, device, criterion_test)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    labels = [ACTIVITY_NAMES[i] for i in range(1, 7)]
    cm = confusion_matrix(y_true0, y_pred0)
    plot_confusion(cm, labels, outdir, name="tcn_confusion_matrix.png", title=f"TCN Confusion Matrix (acc={test_acc:.3f})")

    report = classification_report(y_true0, y_pred0, target_names=labels, digits=4)
    (outdir / "tcn_classification_report.txt").write_text(report, encoding="utf-8")
    print("[SAVE]", outdir / "tcn_classification_report.txt")

    #embedding 可视化
    emb, y0 = extract_embeddings(model, test_loader, device)  # y0: 0..5
    visualize_embeddings(emb, y0 + 1, outdir, prefix="tcn_test", random_state=0)

    # 保存模型
    torch.save(model.state_dict(), outdir / "tcn_best.pt")
    print("[SAVE]", outdir / "tcn_best.pt")

if __name__ == "__main__":
    main()
