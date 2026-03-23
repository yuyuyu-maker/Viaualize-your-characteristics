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

# 配置
model_path = "xr_yolo26.pt"  # 你的训练好的YOLO模型
rendered_dir =  "dataset/images/train"
real_dir     = "dataset_split/images/train"
sample_per_dataset = 3000   # 每个数据集随机抽取数量
output_image = "tsne_visualization.png"  # 保存的可视化图像路径
random.seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载YOLO模型
model = YOLO(model_path)
model.model.eval()
model.model.to(device)

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((640,640)),
    transforms.ToTensor(),
])

def extract_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # [1,3,H,W]
    
    with torch.no_grad():
        # backbone输出：model.model.model[0]通常是backbone
        features = model.model.model[0](x)  # 形状 [1,C,H',W']
        feat_vector = F.adaptive_avg_pool2d(features, 1).flatten().cpu().numpy()  # [C]
    return feat_vector

# 收集每个数据集的图像路径
def list_image_paths(dir_path):
    paths = []
    for img_name in os.listdir(dir_path):
        if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            paths.append(os.path.join(dir_path, img_name))
    return paths

rendered_paths = list_image_paths(rendered_dir)
real_paths = list_image_paths(real_dir)

# 每个数据集随机抽取 sample_per_dataset 张（不足则全取）
rendered_sampled = random.sample(rendered_paths, min(sample_per_dataset, len(rendered_paths)))
real_sampled = random.sample(real_paths, min(sample_per_dataset, len(real_paths)))

print(f"rendered: 抽取 {len(rendered_sampled)} 张 (共 {len(rendered_paths)} 张)")
print(f"real:     抽取 {len(real_sampled)} 张 (共 {len(real_paths)} 张)")

features = []
labels = []
for img_path in rendered_sampled:
    feat = extract_feature(img_path)
    features.append(feat)
    labels.append("rendered")
for img_path in real_sampled:
    feat = extract_feature(img_path)
    features.append(feat)
    labels.append("real")

features = np.array(features)
labels = np.array(labels)

# t-SNE降维到2D
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# 可视化并保存
plt.figure(figsize=(8, 6))
for dataset_name in np.unique(labels):
    idx = labels == dataset_name
    plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=dataset_name, alpha=0.6)
plt.legend()
plt.title("YOLO Backbone Feature t-SNE Visualization")
plt.tight_layout()
plt.savefig(output_image, dpi=150, bbox_inches="tight")
plt.close()
print(f"可视化已保存到: {output_image}")