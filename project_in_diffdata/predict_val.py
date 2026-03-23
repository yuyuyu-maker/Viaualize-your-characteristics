import os
import random
from ultralytics import YOLO

# 配置
model_path = "both_train.pt"           # 模型权重
val_image_dir = "/workspace/all/dataset_split/images/val"    # val 图片目录
max_images = 300                        # 预测数量
save_dir = "true_val_300_results"       # 结果保存文件夹
random.seed(42)                         # 可复现的随机抽样

def main():
    model = YOLO(model_path)

    # 收集图片路径（仅常见图片格式）
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    all_paths = [
        os.path.join(val_image_dir, f)
        for f in os.listdir(val_image_dir)
        if f.lower().endswith(exts)
    ]
    image_list = random.sample(all_paths, min(max_images, len(all_paths)))

    if not image_list:
        print(f"未在 {val_image_dir} 中找到图片")
        return
    print(f"共 {len(all_paths)} 张图片，随机抽取 {len(image_list)} 张进行预测")

    # 保存到指定文件夹：project 为父目录，name 为本次运行子目录
    parent = os.path.dirname(save_dir)
    name = os.path.basename(save_dir)
    if not parent:
        parent = "."
    results = model.predict(
        source=image_list,
        save=True,
        project=parent,
        name=name,
        exist_ok=True,
    )

    out_path = os.path.abspath(save_dir)
    print(f"预测完成，结果已保存到: {out_path}")

if __name__ == "__main__":
    main()
