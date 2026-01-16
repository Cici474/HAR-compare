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
# ============================================================
# 0) 工具：输出目录、保存图像
# ============================================================
#创建输出目录
def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def savefig(outdir: Path, name: str):
    path = outdir / name  #path对象拼接，生成图像保存路径
    plt.tight_layout()  #自动调整子图间距，避免标签重叠
    plt.savefig(path, dpi=220)
    print("[SAVE]", path)

# ============================================================
# 1) 自动定位 UCI HAR 根目录
# ============================================================
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
        "找不到 UCI HAR 根目录。请确保 start_path 位于数据集目录内部。\n"
        "根目录应包含 features.txt、train/X_train.txt、test/X_test.txt"
    )

# ============================================================
# 2) 读取 561 维特征（X_train/X_test）+ 标签
# ============================================================
def load_features_names(root: Path):
    #读取，格式：序号 特征名
    feats = pd.read_csv(root / "features.txt", sep=r"\s+", header=None, names=["idx", "name"])
    # 处理重复特征名，避免命名冲突
    name_counts = {}
    fixed_names = []
    for n in feats["name"].tolist():
        if n not in name_counts:
            name_counts[n] = 0
            fixed_names.append(n)
        else:
            name_counts[n] += 1
            fixed_names.append(f"{n}__dup{name_counts[n]}") #重复名加后缀
    return fixed_names

def load_feature_dataset(root: Path):
    feature_names = load_features_names(root) #获取处理后的特征名
    X_train = np.loadtxt(root / "train" / "X_train.txt")
    y_train = np.loadtxt(root / "train" / "y_train.txt").astype(int)
    X_test  = np.loadtxt(root / "test" / "X_test.txt")
    y_test  = np.loadtxt(root / "test" / "y_test.txt").astype(int)
    return X_train, y_train, X_test, y_test, feature_names

# ============================================================
# 3) 读取 Inertial Signals（时间序列 128×9）
#    UCI HAR inertial signals 文件（train/test）：
#      body_acc_x/y/z
#      body_gyro_x/y/z
#      total_acc_x/y/z
#    每个文件形状：(N, 128)
#    最终拼成：(N, 9, 128)  -> (batch, channels=9, length=128)
# ============================================================
INERTIAL_FILES = [
    "body_acc_x_", "body_acc_y_", "body_acc_z_",
    "body_gyro_x_", "body_gyro_y_", "body_gyro_z_",
    "total_acc_x_", "total_acc_y_", "total_acc_z_",
]
def load_inertial_split(root: Path, split: str):
    base = root / split / "Inertial Signals"
    if not base.exists():
        raise FileNotFoundError(f"找不到 Inertial Signals 目录：{base}")

    mats = []
    for prefix in INERTIAL_FILES:
        p = base / f"{prefix}{split}.txt"
        if not p.exists():
            raise FileNotFoundError(f"缺少 inertial 文件：{p}")
        m = np.loadtxt(p)  # 读取单个通道，形状：(N, 128)-(样本数，时间步长)
        mats.append(m)
    # 拼接9个通道：(9, N, 128) -> (N, 9, 128)[batch, channels, length]
    X = np.stack(mats, axis=0).transpose(1, 0, 2)
    y = np.loadtxt(root / split / f"y_{split}.txt").astype(int)
    return X, y

def load_inertial_dataset(root: Path):
    X_train, y_train = load_inertial_split(root, "train")
    X_test, y_test = load_inertial_split(root, "test")
    return X_train, y_train, X_test, y_test

# ============================================================
# 4) 可视化：类别分布、混淆矩阵、训练曲线、embedding 可视化
# ============================================================
def plot_class_distribution(y, outdir: Path, name="class_distribution.png", title="Class distribution"):
    plt.figure()
    vals = pd.Series(y).value_counts().sort_index()
    plt.bar([ACTIVITY_NAMES[int(i)] for i in vals.index], vals.values)
    plt.xticks(rotation=25, ha="right")
    plt.title(title)
    savefig(outdir, name)
    plt.close()

def plot_confusion_matrix(cm, labels, outdir: Path, name: str, title: str):
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

def plot_curves(history: dict, outdir: Path, prefix: str):
    # history: {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[]}
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

#将模型中间层的嵌入向量降维到2D，可视化类别间的区分度。聚类效果越好，模型特征提取能力越强。
def visualize_embeddings(emb, y, outdir: Path, prefix: str, random_state=0):
    # emb: (N, D)
    y = np.asarray(y)
    # PCA降维到2D(线性降维，保留全局结构)
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
    # t-SNE 2D（非线性降维，更适合局部聚类）
    tsne = TSNE(
        n_components=2,
        perplexity=35, #适配样本数的领域大小
        learning_rate="auto",
        init="pca", #用PCA初始化，加速收敛且更稳定
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

# ============================================================
# 5) 封装numpy数据为pytorch可迭代的dataset类
# ============================================================
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # label: 1..6 -> 0..5
        self.y = torch.tensor(y - 1, dtype=torch.long)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================
# 6) 模型：MLP（561维特征）
# ============================================================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim=561, num_classes=6):
        super().__init__()
        #提取特征
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256), #线性层：561维->256维
            nn.ReLU(), #激活函数：引入非线性
            nn.Dropout(0.2), #随机失活：防止过拟合
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Linear(128, num_classes) #分类头：128维嵌入 -> 6类预测

    def forward(self, x, return_emb=False):
        emb = self.backbone(x)
        logits = self.head(emb) #未归一化的预测值
        if return_emb:
            return logits, emb
        return logits

# ============================================================
# 7) 模型：1D-CNN（时间序列 9×128）
# ============================================================
class CNN1DClassifier(nn.Module):
    def __init__(self, in_ch=9, num_classes=6):
        super().__init__()
        #特征提取主干：1D卷积(处理时间序列)
        self.features = nn.Sequential(
            #第一层卷积 9->64
            nn.Conv1d(in_ch, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), #拟归一化：加速训练，稳定梯度
            nn.ReLU(),
            nn.MaxPool1d(2),  # 池化：长度128 -> 64
            #第二层卷积 64->128
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 64 -> 32

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 自适应池化-> (B, 128, 1)
        )
        self.fc_emb = nn.Linear(128, 128)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x, return_emb=False):
        # x: (B, 9, 128)
        h = self.features(x).squeeze(-1)  # 池化后(B, 128)
        emb = F.relu(self.fc_emb(h))
        logits = self.head(emb)
        if return_emb:
            return logits, emb
        return logits

# ============================================================
# 8) 训练/评估
# ============================================================
def accuracy_from_logits(logits, y):
    pred = torch.argmax(logits, dim=1)
    return (pred == y).float().mean().item()

@torch.no_grad() #无梯度计算，节省内存+加速
def eval_model(model, loader, device):
    model.eval()
    losses = []
    accs = []
    all_pred = []
    all_true = []
    ce = nn.CrossEntropyLoss() #交叉熵损失(分类任务)

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        logits = model(X) #前向传播
        loss = ce(logits, y).item()
        acc = accuracy_from_logits(logits, y)

        losses.append(loss)
        accs.append(acc)
        #收集
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        all_pred.append(pred)
        all_true.append(y.cpu().numpy())
#拼接
    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)
    return float(np.mean(losses)), float(np.mean(accs)), all_true, all_pred

def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3, weight_decay=1e-4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val = -1.0 #记录最佳验证集的acc
    best_state = None #记录最佳模型参数

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_accs = []

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits = model(X)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            train_losses.append(loss.item())
            train_accs.append(accuracy_from_logits(logits, y))

        val_loss, val_acc, _, _ = eval_model(model, val_loader, device)

        history["train_loss"].append(float(np.mean(train_losses)))
        history["train_acc"].append(float(np.mean(train_accs)))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[Epoch {ep:02d}] train_loss={history['train_loss'][-1]:.4f} "
              f"train_acc={history['train_acc'][-1]:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
#保存最佳
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    embs = []
    ys = []
    for X, y in loader:
        X = X.to(device)
        logits, emb = model(X, return_emb=True)
        embs.append(emb.cpu().numpy())
        ys.append(y.cpu().numpy())
    return np.vstack(embs), np.concatenate(ys)

# ============================================================
# 9) 方案A：561维特征 + MLP（快且稳定）
# ============================================================
def run_mlp_pipeline(root: Path, outdir: Path, device):
    X_train, y_train, X_test, y_test, feature_names = load_feature_dataset(root)
    # EDA：类别分布
    plot_class_distribution(y_train, outdir, name="mlp_train_class_dist.png", title="Train class distribution (feature-based)")
    # 特征标准化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # train 划分 val（从 train 中划出一部分做验证）
    n = X_train_s.shape[0]
    idx = np.arange(n)
    np.random.seed(0) #固定随机种子，保证可复现
    np.random.shuffle(idx)
    val_size = int(0.2 * n)
    val_idx = idx[:val_size]
    tr_idx  = idx[val_size:]

    X_tr, y_tr = X_train_s[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train_s[val_idx], y_train[val_idx]

    train_loader = DataLoader(NumpyDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    val_loader   = DataLoader(NumpyDataset(X_val, y_val), batch_size=512, shuffle=False)
    test_loader  = DataLoader(NumpyDataset(X_test_s, y_test), batch_size=512, shuffle=False)
    #初始化+训练模型
    model = MLPClassifier(in_dim=X_train_s.shape[1], num_classes=6)
    model, history = train_model(model, train_loader, val_loader, device, epochs=25, lr=1e-3)

    plot_curves(history, outdir, prefix="mlp")

    # 测试评估
    test_loss, test_acc, y_true0, y_pred0 = eval_model(model, test_loader, device)
    print(f"[MLP TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    labels = [ACTIVITY_NAMES[i] for i in range(1, 7)]
    cm = confusion_matrix(y_true0, y_pred0)
    plot_confusion_matrix(cm, labels, outdir, name="mlp_confusion_matrix.png", title=f"MLP Confusion Matrix (acc={test_acc:.3f})")

    report = classification_report(y_true0, y_pred0, target_names=labels, digits=4)
    (outdir / "mlp_classification_report.txt").write_text(report, encoding="utf-8")
    print("[SAVE]", outdir / "mlp_classification_report.txt")

    # embedding 可视化
    emb, y0 = extract_embeddings(model, test_loader, device)
    visualize_embeddings(emb, y0 + 1, outdir, prefix="mlp_test", random_state=0)

# ============================================================
# 10) 方案B：Inertial Signals + 1D-CNN
# ============================================================
def run_cnn_pipeline(root: Path, outdir: Path, device):
    X_train, y_train, X_test, y_test = load_inertial_dataset(root)

    print("[INFO] Inertial shapes:")
    print("  X_train:", X_train.shape, " (should be N, 9, 128)")
    print("  X_test :", X_test.shape)

    plot_class_distribution(y_train, outdir, name="cnn_train_class_dist.png", title="Train class distribution (inertial signals)")

    # 归一化（按通道做标准化：对每个 channel 统计 train 的 mean/std）
    # 形状：(N, 9, 128)
    ch_mean = X_train.mean(axis=(0, 2), keepdims=True)  # (1, 9, 1)
    ch_std  = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train_n = (X_train - ch_mean) / ch_std
    X_test_n  = (X_test  - ch_mean) / ch_std

    # train 划 val
    n = X_train_n.shape[0]
    idx = np.arange(n)
    np.random.seed(0)
    np.random.shuffle(idx)
    val_size = int(0.2 * n)
    val_idx = idx[:val_size]
    tr_idx  = idx[val_size:]

    X_tr, y_tr = X_train_n[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train_n[val_idx], y_train[val_idx]

    train_loader = DataLoader(NumpyDataset(X_tr, y_tr), batch_size=256, shuffle=True)
    val_loader   = DataLoader(NumpyDataset(X_val, y_val), batch_size=512, shuffle=False)
    test_loader  = DataLoader(NumpyDataset(X_test_n, y_test), batch_size=512, shuffle=False)

    model = CNN1DClassifier(in_ch=9, num_classes=6)
    model, history = train_model(model, train_loader, val_loader, device, epochs=30, lr=8e-4)

    plot_curves(history, outdir, prefix="cnn1d")

    # 测试评估
    test_loss, test_acc, y_true0, y_pred0 = eval_model(model, test_loader, device)
    print(f"[CNN TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    labels = [ACTIVITY_NAMES[i] for i in range(1, 7)]
    cm = confusion_matrix(y_true0, y_pred0)
    plot_confusion_matrix(cm, labels, outdir, name="cnn1d_confusion_matrix.png", title=f"1D-CNN Confusion Matrix (acc={test_acc:.3f})")

    report = classification_report(y_true0, y_pred0, target_names=labels, digits=4)
    (outdir / "cnn1d_classification_report.txt").write_text(report, encoding="utf-8")
    print("[SAVE]", outdir / "cnn1d_classification_report.txt")

    # embedding 可视化（test）
    emb, y0 = extract_embeddings(model, test_loader, device)
    visualize_embeddings(emb, y0 + 1, outdir, prefix="cnn1d_test", random_state=0)


def main():
    START_PATH = r"E:\yan\class\data data\human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\Inertial Signals"

    outdir = ensure_outdir(Path("./outputs_dl"))
    root = find_ucihar_root(START_PATH)
    print("[INFO] UCI HAR root =", root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device =", device)
    # 方案A：561维特征 + MLP
    print("\n===== (A) Feature-based MLP =====")
    run_mlp_pipeline(root, outdir, device)
    # 方案B：原始时间序列 + 1D-CNN
    print("\n===== (B) Inertial signals 1D-CNN =====")
    run_cnn_pipeline(root, outdir, device)


if __name__ == "__main__":
    main()
