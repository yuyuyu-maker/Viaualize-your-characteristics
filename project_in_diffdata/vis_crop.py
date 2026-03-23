"""
基于装甲板裁剪的 t-SNE 可视化：
从 xr(渲染) / true(真实) 的 images 读图，用对应 labels 中第 2–5 列 (xywh) 扣出装甲板（略放大），
对每个扣图做 YOLO backbone 特征提取，再 t-SNE 可视化并保存。

支持「逐层」提取：对 backbone 的每一层分别提特征、做 t-SNE，便于排查浅层/深层差异。
"""
import os
import random
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 配置：xr = 渲染，true = 真实（red3/xr 与 red3/true，下含 images / labels）
red3_root = "/workspace/all/red3"
xr_image_dir = os.path.join(red3_root, "xr", "images")
true_image_dir = os.path.join(red3_root, "true", "images")
xr_label_dir = None
true_label_dir = None

model_path = "test_yolo26.pt"

sample_per_dataset = 1000
expand_ratio = 1.2
# 逐层可视化：提取 backbone 哪些层的特征并分别做 t-SNE（YOLO26 backbone 一般为 0–10）
# 例如 [0, 1, 2, 5, 10] 只看部分层；range(11) 或 None 表示全部 11 层
BACKBONE_LAYERS = None   # None = 所有层 0..10；或设为 [0, 2, 5, 10] 等
output_dir = "./tsne_visualization_crop_layer_with_true"        # 逐层图保存目录；单层时仍可用下方 output_image
output_image = "./tsne_visualization_crop_normal_with_true.png"
random.seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 YOLO 模型
model = YOLO(model_path)
model.model.eval()
model.model.to(device)

preprocess = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])


def get_label_dir(image_dir, explicit_label_dir):
    """由 .../xxx/images 推导 .../xxx/labels（支持路径以 /images 结尾、无尾部斜杠）。"""
    if explicit_label_dir is not None:
        return explicit_label_dir
    d = os.path.normpath(image_dir)
    if d.endswith(os.sep + "images") or d.endswith("images"):
        return os.path.join(os.path.dirname(d), "labels")
    return image_dir.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)


def image_path_to_label_path(img_path, image_dir, label_dir=None):
    rel = os.path.relpath(img_path, image_dir)
    base, _ = os.path.splitext(rel)
    if label_dir is None:
        label_dir = get_label_dir(image_dir, None)
    return os.path.join(label_dir, base + ".txt")


def parse_xywh_lines(label_path, img_w, img_h):
    """
    解析 YOLO 格式 label：每行 class_id x_center y_center width height [keypoints...]
    第 2–5 列（0-indexed 为 1,2,3,4）为归一化 xywh。返回像素坐标的 (x_center, y_center, w, h) 列表。
    """
    if not os.path.isfile(label_path):
        return []
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            xc = float(parts[1])
            yc = float(parts[2])
            w  = float(parts[3])
            h  = float(parts[4])
            # 归一化 -> 像素
            xc_px = xc * img_w
            yc_px = yc * img_h
            w_px  = w * img_w
            h_px  = h * img_h
            boxes.append((xc_px, yc_px, w_px, h_px))
    return boxes


def expand_box(xc, yc, w, h, img_w, img_h, ratio):
    """在保持中心不变下将宽高放大 ratio 倍，并裁剪到图像内，返回 (x1,y1,x2,y2) 整数像素。"""
    w2 = w * ratio / 2
    h2 = h * ratio / 2
    x1 = max(0, int(xc - w2))
    y1 = max(0, int(yc - h2))
    x2 = min(img_w, int(xc + w2))
    y2 = min(img_h, int(yc + h2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def crop_armor_regions(img_path, label_path, expand_ratio):
    """
    读取图片和标签，按每行 xywh 扣出装甲板（略放大），返回 [PIL.Image, ...]。
    无标签或解析不到框则返回空列表。
    """
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


# Backbone 层数（YOLO26n-pose 为 0–10，共 11 层）
NUM_BACKBONE_LAYERS = 11


def extract_features_per_layer(pil_img, layer_indices):
    """
    对一张 PIL 图做预处理，顺序经过 backbone 的 0..k 层，返回指定各层的特征向量。
    backbone 0–10 为顺序结构（每层 from -1），故可逐层 forward。
    layer_indices: 要提取的层下标列表，如 [0, 2, 5, 10]。返回 dict: layer_idx -> (C,) numpy
    """
    x = preprocess(pil_img).unsqueeze(0).to(device)
    out = x
    layer_indices = sorted(set(layer_indices))
    result = {}
    with torch.no_grad():
        for i in range(NUM_BACKBONE_LAYERS):
            out = model.model.model[i](out)
            if i in layer_indices:
                result[i] = F.adaptive_avg_pool2d(out, 1).flatten().cpu().numpy()
    return result


def extract_feature_from_pil(pil_img, layer=0):
    """对一张 PIL 图像提取指定 backbone 层的特征向量（默认第 0 层，兼容原逻辑）。"""
    d = extract_features_per_layer(pil_img, [layer])
    return d[layer]


def list_image_paths(dir_path):
    paths = []
    for f in os.listdir(dir_path):
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            paths.append(os.path.join(dir_path, f))
    return paths


xr_label_dir_resolved = get_label_dir(xr_image_dir, xr_label_dir)
true_label_dir_resolved = get_label_dir(true_image_dir, true_label_dir)

xr_paths = list_image_paths(xr_image_dir)
true_paths = list_image_paths(true_image_dir)
xr_sampled = random.sample(xr_paths, min(sample_per_dataset, len(xr_paths)))
true_sampled = random.sample(true_paths, min(sample_per_dataset, len(true_paths)))

# 要可视化的层：None = 全部 0..NUM_BACKBONE_LAYERS-1
layer_indices = list(range(NUM_BACKBONE_LAYERS)) if BACKBONE_LAYERS is None else list(BACKBONE_LAYERS)
layer_indices = [int(i) for i in layer_indices]

# 逐层收集特征（一次 forward 得到所有层，避免重复计算）
labels_list = []
features_by_layer = {i: [] for i in layer_indices}

for img_path in xr_sampled:
    label_path = image_path_to_label_path(img_path, xr_image_dir, xr_label_dir_resolved)
    crops = crop_armor_regions(img_path, label_path, expand_ratio)
    for crop in crops:
        layer_feats = extract_features_per_layer(crop, layer_indices)
        for i in layer_indices:
            features_by_layer[i].append(layer_feats[i])
        labels_list.append("xr")

for img_path in true_sampled:
    label_path = image_path_to_label_path(img_path, true_image_dir, true_label_dir_resolved)
    crops = crop_armor_regions(img_path, label_path, expand_ratio)
    for crop in crops:
        layer_feats = extract_features_per_layer(crop, layer_indices)
        for i in layer_indices:
            features_by_layer[i].append(layer_feats[i])
        labels_list.append("true")

if not labels_list:
    print("未得到任何装甲板样本，请检查图片路径与标签路径、标签格式（第2–5列为 xywh）。")
    exit(1)

labels = np.array(labels_list)
n_xr = np.sum(labels == "xr")
n_true = np.sum(labels == "true")
print(f"xr(渲染) 装甲板样本: {n_xr}")
print(f"true(真实) 装甲板样本: {n_true}")
print(f"总样本数: {len(labels)}")
print(f"逐层 t-SNE：层 {layer_indices}")

os.makedirs(output_dir, exist_ok=True)

for layer_i in layer_indices:
    features = np.array(features_by_layer[layer_i])
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    for name in np.unique(labels):
        idx = labels == name
        plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=name, alpha=0.6)
    plt.legend()
    plt.title(f"YOLO Backbone Layer {layer_i} t-SNE (Armor Crop): xr vs true")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"tsne_visualization_crop_layer{layer_i}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  层 {layer_i} -> {out_path}")

# 若只看了单层且为第 0 层，额外复制到 output_image 方便兼容旧脚本
if len(layer_indices) == 1 and layer_indices[0] == 0:
    import shutil
    src = os.path.join(output_dir, "tsne_visualization_crop_layer0.png")
    if os.path.abspath(src) != os.path.abspath(output_image):
        shutil.copy(src, output_image)
        print(f"已另存为: {output_image}")
