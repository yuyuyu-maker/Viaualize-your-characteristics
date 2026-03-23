import os
import random
import sys
import torch
from torchvision import transforms
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# ========== 本地加载配置（无需联网）==========
# 在你电脑上下载并拷贝到本机后，填写下面两个路径即可：
#  1. DINO 仓库：git clone https://github.com/facebookresearch/dino.git 得到文件夹路径
#  2. 预训练权重 .pth，按模型选一个下载：
#     dino_vits16: https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth
#     dino_vits8:  https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth
#     dino_vitb16: https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
#     dino_vitb8:  https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth
dino_repo_dir = "dino"                    # 本地 DINO 仓库根目录（含 vision_transformer.py、utils.py）
dino_checkpoint_path = "dino_deitsmall16_pretrain.pth"  # 本地 .pth 权重路径

# 数据路径：xr = 渲染（red3/xr）；true = 真实（dataset_split）
red3_root = "/workspace/all/red3"
xr_image_dir = os.path.join(red3_root, "xr", "images")
dataset_split_root = "/workspace/all/dataset_split"
true_image_dir = os.path.join(dataset_split_root, "images", "train")  # 可改为 images/val
xr_label_dir = None
true_label_dir = None
sample_per_dataset = 3000   # 每个数据集最多参与抽样的图片数
expand_ratio = 1.2           # 装甲板框相对标签框的放大比例
output_image = "tsne_visualization_dino_crop.png"  # 保存的可视化图像路径
dino_model_name = "dino_vits16"
# xr–true 最相似对：保存多少对、保存到哪
num_similar_pairs = 30
pairs_save_dir = "dino_similar_pairs"
random.seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 从本地 DINO 仓库加载模型（不联网）
def _load_dino_local(repo_dir, checkpoint_path, model_name):
    repo_dir = os.path.abspath(repo_dir)
    if not os.path.isdir(repo_dir):
        raise FileNotFoundError(f"DINO 仓库目录不存在: {repo_dir}\n请 clone: git clone https://github.com/facebookresearch/dino.git")

    # 权重路径：先按原路径找，再试「相对脚本目录」「相对仓库目录」
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        checkpoint_path,
        os.path.join(script_dir, checkpoint_path),
        os.path.join(repo_dir, checkpoint_path),
        os.path.join(script_dir, os.path.basename(checkpoint_path)),
    ]
    checkpoint_path = None
    for p in candidates:
        if os.path.isfile(p):
            checkpoint_path = os.path.abspath(p)
            break
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"权重文件未找到。已尝试: {candidates}\n"
            "请下载对应 .pth 后放到本机，并修改 dino_checkpoint_path（可写相对 vis_dino.py 的路径或相对 dino 仓库的路径）。"
        )
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import vision_transformer as vits

    arch_map = {
        "dino_vits16": ("vit_small", 16),
        "dino_vits8": ("vit_small", 8),
        "dino_vitb16": ("vit_base", 16),
        "dino_vitb8": ("vit_base", 8),
    }
    if model_name not in arch_map:
        raise ValueError(f"不支持的模型: {model_name}，可选: {list(arch_map.keys())}")
    arch, patch_size = arch_map[model_name]
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model

print(f"从本地加载 DINO: {dino_model_name} (仓库={dino_repo_dir}, 权重={dino_checkpoint_path}) ...")
model = _load_dino_local(dino_repo_dir, dino_checkpoint_path, dino_model_name)
model.eval()
model.to(device)

# DINO 使用 ImageNet 标准预处理，输入 224x224
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def image_path_to_label_path(img_path, image_dir, label_dir=None):
    rel = os.path.relpath(img_path, image_dir)
    base, _ = os.path.splitext(rel)
    if label_dir is None:
        label_dir = get_label_dir(image_dir, None)
    return os.path.join(label_dir, base + ".txt")


def parse_xywh_lines(label_path, img_w, img_h):
    if not os.path.isfile(label_path):
        return []
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or len(line.split()) < 5:
                continue
            parts = line.split()
            xc = float(parts[1])
            yc = float(parts[2])
            w, h = float(parts[3]), float(parts[4])
            boxes.append((xc * img_w, yc * img_h, w * img_w, h * img_h))
    return boxes


def expand_box(xc, yc, w, h, img_w, img_h, ratio):
    w2, h2 = w * ratio / 2, h * ratio / 2
    x1 = max(0, int(xc - w2))
    y1 = max(0, int(yc - h2))
    x2 = min(img_w, int(xc + w2))
    y2 = min(img_h, int(yc + h2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_armor_regions(img_path, label_path, expand_ratio):
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    boxes = parse_xywh_lines(label_path, img_w, img_h)
    crops = []
    for xc, yc, w, h in boxes:
        box = expand_box(xc, yc, w, h, img_w, img_h, expand_ratio)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        crops.append(img.crop((x1, y1, x2, y2)))
    return crops


def extract_feature_from_pil(pil_img):
    x = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        if out.dim() == 3:
            out = out[:, 0, :]
        feat_vector = out.flatten().cpu().numpy()
    return feat_vector


def list_image_paths(dir_path):
    paths = []
    for f in os.listdir(dir_path):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            paths.append(os.path.join(dir_path, f))
    return paths


def get_label_dir(image_dir, explicit_label_dir):
    """由 .../xxx/images 推导 .../xxx/labels（支持路径以 /images 结尾、无尾部斜杠）。"""
    if explicit_label_dir is not None:
        return explicit_label_dir
    d = os.path.normpath(image_dir)
    if d.endswith(os.sep + "images") or d.endswith("images"):
        return os.path.join(os.path.dirname(d), "labels")
    return image_dir.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)


xr_label_dir_resolved = get_label_dir(xr_image_dir, xr_label_dir)
true_label_dir_resolved = get_label_dir(true_image_dir, true_label_dir)
xr_paths = list_image_paths(xr_image_dir)
true_paths = list_image_paths(true_image_dir)
xr_sampled = random.sample(xr_paths, min(sample_per_dataset, len(xr_paths)))
true_sampled = random.sample(true_paths, min(sample_per_dataset, len(true_paths)))

# 分开存：用于 t-SNE 的 features/labels，以及用于找相似对的 (特征, crop, 原图文件名)
features = []
labels = []
xr_features = []
xr_crops = []
xr_src_names = []
true_features = []
true_crops = []
true_src_names = []

for img_path in xr_sampled:
    label_path = image_path_to_label_path(img_path, xr_image_dir, xr_label_dir_resolved)
    crops = crop_armor_regions(img_path, label_path, expand_ratio)
    src_name = os.path.basename(img_path)
    for crop in crops:
        feat = extract_feature_from_pil(crop)
        features.append(feat)
        labels.append("xr")
        xr_features.append(feat)
        xr_crops.append(crop)
        xr_src_names.append(src_name)
for img_path in true_sampled:
    label_path = image_path_to_label_path(img_path, true_image_dir, true_label_dir_resolved)
    crops = crop_armor_regions(img_path, label_path, expand_ratio)
    src_name = os.path.basename(img_path)
    for crop in crops:
        feat = extract_feature_from_pil(crop)
        features.append(feat)
        labels.append("true")
        true_features.append(feat)
        true_crops.append(crop)
        true_src_names.append(src_name)

if not features:
    print("未得到任何装甲板样本，请检查图片路径与标签路径、标签格式（第2–5列为 xywh）。")
    exit(1)

features = np.array(features)
labels = np.array(labels)
print(f"xr(渲染) 装甲板样本: {np.sum(labels == 'xr')}")
print(f"true(真实) 装甲板样本: {np.sum(labels == 'true')}")
print(f"总样本数: {len(features)}")

# t-SNE 降维到 2D
print("运行 t-SNE ...")
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# 可视化并保存
plt.figure(figsize=(8, 6))
for dataset_name in np.unique(labels):
    idx = labels == dataset_name
    plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=dataset_name, alpha=0.6)
plt.legend()
plt.title("DINO Feature t-SNE (Armor Crop): xr vs true")
plt.tight_layout()
plt.savefig(output_image, dpi=150, bbox_inches="tight")
plt.close()
print(f"可视化已保存到: {output_image}")

# 找 true–xr 最相似的 N 对（每张图不重复），按「一对一个文件夹」保存
if xr_features and true_features and num_similar_pairs > 0:
    R = np.array(xr_features)
    Re = np.array(true_features)
    dist = np.sqrt(((R[:, None, :] - Re[None, :, :]) ** 2).sum(axis=2))
    n_xr, n_true = dist.shape
    order = np.argsort(dist.ravel())
    used_xr, used_true = set(), set()
    pairs = []
    for idx in order:
        i, j = idx // n_true, idx % n_true
        if i in used_xr or j in used_true:
            continue
        used_xr.add(i)
        used_true.add(j)
        pairs.append((i, j, float(dist[i, j])))
        if len(pairs) >= num_similar_pairs:
            break
    os.makedirs(pairs_save_dir, exist_ok=True)
    for rank, (i, j, d) in enumerate(pairs, start=1):
        pair_dir = os.path.join(pairs_save_dir, f"pair_{rank:03d}")
        os.makedirs(pair_dir, exist_ok=True)
        xr_crops[i].save(os.path.join(pair_dir, "xr.jpg"), quality=95)
        true_crops[j].save(os.path.join(pair_dir, "true.jpg"), quality=95)
        with open(os.path.join(pair_dir, "info.txt"), "w") as f:
            f.write(f"distance={d:.4f}\n")
            f.write(f"xr={xr_src_names[i]}\n")
            f.write(f"true={true_src_names[j]}\n")
    print(f"最相似 {len(pairs)} 对（无重复）已保存到: {os.path.abspath(pairs_save_dir)} (pair_001 ~ pair_{len(pairs):03d})")
