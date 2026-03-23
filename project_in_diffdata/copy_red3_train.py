"""
从 dataset_split 的 train 集中，只选取「文件名里包含 red3」的图片（不读标签类别），
随机抽取若干张，将图片与对应标签一并复制到 red3/true/ 下（images + labels 结构）。
"""
import os
import random
import shutil

# 配置
#这里你记得自己改掉，我懒得写出那种读取传参数的代码（）
dataset_root = "/workspace/all/dataset_split"
images_split = "images/train"
labels_split = "labels/train"
NAME_KEYWORD = "red3"  
num_copy = 480
output_root = "/workspace/all/red3/true"
random.seed(42)


def main():
    img_dir = os.path.join(dataset_root, images_split)
    lab_dir = os.path.join(dataset_root, labels_split)
    out_img = os.path.join(output_root, "images")
    out_lab = os.path.join(output_root, "labels")
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    kw = NAME_KEYWORD.lower()

    candidates = []  # (image_path, label_path)
    for fname in os.listdir(img_dir):
        lower = fname.lower()
        if not any(lower.endswith(e) for e in exts):
            continue
        if kw not in lower:
            continue
        base, _ = os.path.splitext(fname)
        label_path = os.path.join(lab_dir, base + ".txt")
        if not os.path.isfile(label_path):
            continue
        img_path = os.path.join(img_dir, fname)
        candidates.append((img_path, label_path))

    if not candidates:
        print(f"未在 {img_dir} 中找到文件名含「{NAME_KEYWORD}」且标签存在的样本")
        return

    n = min(num_copy, len(candidates))
    picked = random.sample(candidates, n)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lab, exist_ok=True)

    for img_path, label_path in picked:
        name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(out_img, name))
        shutil.copy2(label_path, os.path.join(out_lab, os.path.basename(label_path)))

    print(f"文件名含「{NAME_KEYWORD}」的样本共 {len(candidates)} 对，已随机复制 {n} 对到:")
    print(f"  图片: {os.path.abspath(out_img)}")
    print(f"  标签: {os.path.abspath(out_lab)}")


if __name__ == "__main__":
    main()
