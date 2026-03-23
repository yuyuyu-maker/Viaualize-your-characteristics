# Viaualize-your-characteristics

本项目围绕 **YOLO 目标检测（含姿态/装甲板任务）** 与 **渲染图（xr）/ 真实图（true）数据域差异** 的分析与可视化：用全图或按标签裁剪的装甲板区域提取特征，再通过 **t-SNE** 观察两类数据在特征空间中的分布；部分脚本还包含验证集抽样推理、子集数据导出等辅助流程。

主要可执行代码位于目录 **`project_in_diffdata/`** 下；`dino/` 为 Facebook [DINO](https://github.com/facebookresearch/dino) 官方仓库的本地拷贝，供 `vis_dino.py` 离线加载 ViT 与预训练权重使用。

---

## 环境依赖

- **Python 3**
- **PyTorch**（建议带 CUDA，脚本会自动选择 `cuda` / `cpu`）
- **ultralytics**（YOLO：`predict_val.py`、`vis.py`、`vis_crop.py`）
- **torchvision**、**PIL**
- **scikit-learn**（`TSNE`）
- **matplotlib**、**numpy**

安装示例（按你当前环境调整）：

```bash
pip install ultralytics torch torchvision pillow scikit-learn matplotlib numpy
```

---

## 目录结构（与本 README 相关部分）

| 路径 | 说明 |
|------|------|
| `project_in_diffdata/` | 本项目自研脚本与输出目录 |
| `project_in_diffdata/dino/` | DINO 源码（`vision_transformer.py` 等），**不修改也可**仅作 import 路径 |
| `project_in_diffdata/dino_similar_pairs/` | `vis_dino.py` 运行后生成的「xr–true 最相似对」示例图与 `info.txt`（若已生成） |

---

## 脚本说明（按职责）

以下路径均相对于 **`project_in_diffdata/`**，运行前请在该目录下执行，或注意修改脚本中的 **绝对路径**（当前仓库内多处写死了 `/workspace/all/...`，换机器需改）。

### 1. `predict_val.py` — 验证集随机抽样推理

**作用：** 加载训练好的 YOLO 权重，从 **验证集图片目录** 中随机抽取若干张，调用 `model.predict` 做推理，并把 **带框/可视化结果** 保存到指定子目录（Ultralytics 的 `project` + `name` 约定）。

**你需要关心的配置（文件顶部）：**

| 变量 | 含义 |
|------|------|
| `model_path` | 权重文件，如 `both_train.pt` |
| `val_image_dir` | 验证集图片根目录 |
| `max_images` | 最多预测多少张（随机抽样） |
| `save_dir` | 结果保存目录名（相对当前工作目录的 `project/name` 结构） |

**适用场景：** 快速抽查模型在 val 上的可视化效果，固定随机种子便于复现同一次抽样。

---

### 2. `copy_red3_train.py` — 按文件名关键字筛选并复制 train 子集

**作用：** 在 `dataset_split` 的 **train** 划分中，只保留 **文件名包含指定关键字**（默认 `red3`）的图片；若存在同名 **YOLO 标签**（`.txt`），则将 **图片 + 标签** 成对随机复制到输出目录，形成标准 `images/`、`labels/` 结构。  
**不读取标签里的类别**，仅按文件名过滤。

**典型用途：** 从大数据集中抠出一批「与 red3 相关」的样本，放到例如 `red3/true/` 供后续与渲染集 `red3/xr/` 对比。

**关键配置：** `dataset_root`、`images_split`、`labels_split`、`NAME_KEYWORD`、`num_copy`、`output_root`。

---

### 3. `vis.py` — 全图 YOLO Backbone 特征的 t-SNE

**作用：** 从两个目录各随机抽取若干张 **整图**（不做裁剪），用 **当前 YOLO 模型的 backbone**（脚本里为 `model.model.model[0](x)` 的输出经全局平均池化得到向量）提取特征，对两类样本一起做 **t-SNE 降到 2 维**，保存散点图 PNG。

**两类数据在脚本中的语义：**

- `rendered_dir`：例如渲染/合成数据（脚本里默认 `dataset/images/train`）
- `real_dir`：例如真实采集或划分后的 train（默认 `dataset_split/images/train`）

**输出：** `output_image` 指定的 PNG（默认 `tsne_visualization.png`）。

**适用场景：** 粗看「渲染 vs 真实」在 **自己训练的 YOLO 特征** 下是否可分、分布是否重叠。

---

### 4. `vis_crop.py` — 装甲板裁剪 + YOLO 多 Backbone 层 t-SNE

**作用：**  
- 从 `red3/xr/images` 与 `red3/true/images`（路径可改）读图；  
- 根据 **YOLO 格式标签** 第 2–5 列（归一化 **xywh**）在图上裁 **装甲板区域**，并可按 `expand_ratio` 略放大框；  
- 对每个裁剪块，沿 YOLO backbone **逐层**（默认 11 层，对应 `NUM_BACKBONE_LAYERS`）提取特征，**每一层单独做一次 t-SNE**，输出多张图到 `output_dir`。

**与 `vis.py` 的区别：** 只看装甲板局部、且可对比 **浅层/深层** 上「xr vs true」分离情况。

**关键配置：** `red3_root`、`model_path`、`sample_per_dataset`、`expand_ratio`、`BACKBONE_LAYERS`（`None` 表示全部层，或设为 `[0,2,5,10]` 等子集）、`output_dir` / `output_image`。

---

### 5. `vis_dino.py` — DINO 预训练特征下的装甲板 t-SNE + 最相似 xr–true 对

**作用：**  
1. **离线**加载本地 `dino` 仓库中的 ViT 与 **本地下载的 `.pth` 权重**（无需联网推理）。  
2. 与 `vis_crop.py` 类似，按标签裁装甲板，用 **DINO**（默认 `dino_vits16`）提取 **CLS/全局特征**（224×224、ImageNet 归一化）。  
3. 对 **xr** 与 **true** 特征做 t-SNE，保存 `tsne_visualization_dino_crop.png`。  
4. 在特征空间用 **欧氏距离** 做 **xr 与 true  crop 之间的最近邻匹配**，在尽量不重复配对的前提下，保存 **前 N 对** 到 `pairs_save_dir`（如 `dino_similar_pairs/pair_001/`，内含 `xr.jpg`、`true.jpg`、`info.txt`）。

**前置条件：**  
- `dino_repo_dir` 指向含 `vision_transformer.py` 的目录；  
- `dino_checkpoint_path` 指向已下载的预训练权重（脚本注释中有官方下载链接说明）。

**关键配置：** `red3_root`、`dataset_split_root`（真实图所在根）、`true_image_dir`（可改为 `images/val` 等）、`sample_per_dataset`、`num_similar_pairs`、`pairs_save_dir`。

**适用场景：** 用 **通用自监督视觉特征** 观察域差异；并定性查看「渲染图与哪张真实裁剪最像」。

---

## 数据与标签约定

- **YOLO 检测/姿态标签：** 每行至少 5 列：`class x_center y_center width height`（归一化 0–1）。裁剪脚本使用 **第 2–5 列** 作为框。若你还有关键点等额外列，只要前 5 列符合上述格式即可。  
- **路径习惯：** 图片在 `.../images/`，标签在同级 `.../labels/`， basename 一致（`.jpg` ↔ `.txt`）。

---

## 运行方式示例

在配置好权重与数据路径后：

```bash
cd project_in_diffdata

# 验证集抽样预测
python predict_val.py

# 复制文件名含 red3 的 train 样本
python copy_red3_train.py

# 全图 YOLO 特征 t-SNE
python vis.py

# 装甲板裁剪 + 逐层 YOLO t-SNE
python vis_crop.py

# DINO 特征 t-SNE + 相似对导出（需本地 dino 与 .pth）
python vis_dino.py
```

---

## 子模块 `project_in_diffdata/dino/`

此为 Meta FAIR **DINO** 官方代码，用于自监督 ViT 训练与评估。本仓库中 **仅 `vis_dino.py` 依赖其 Python 模块与权重文件**；该目录自带的 `README.md`、`LICENSE` 以原项目为准。

---

## 说明

- 脚本中的 **绝对路径**（如 `/workspace/all/dataset_split`、`/workspace/all/red3`）为作者环境示例，克隆到其他机器后请统一修改。  
- 根目录项目名 **Viaualize** 为历史拼写，若需与「Visualize」统一可在后续重命名仓库时一并更正。
