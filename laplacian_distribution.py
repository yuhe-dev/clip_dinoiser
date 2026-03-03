import os
import cv2
import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------------------------------------------------------------------
# 1. 配置路径与类别定义 (使用你提供的列表)
# --------------------------------------------------------------------------------
IMG_DIR = "data/coco_stuff164k/images/train2017"
ANN_DIR = "data/coco_stuff164k/annotations/train2017" # 存放 _labelTrainIds.png 的目录

CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
    'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building-other', 'bush', 
    'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 'cloth', 
    'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 
    'door-stuff', 'fence', 'floor-marble', 'floor-other', 'floor-stone', 'floor-tile', 
    'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 
    'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 
    'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 
    'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 
    'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 
    'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw', 
    'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 
    'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone', 'wall-tile', 
    'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood'
)

# --------------------------------------------------------------------------------
# 2. 采样与切片逻辑
# --------------------------------------------------------------------------------
def get_sharpness(path):
    img = cv2.imread(path)
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def analyze_subsets():
    all_imgs = [f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')]
    random.seed(42)
    pool = random.sample(all_imgs, 20000)
    
    print("Scoring 20k images...")
    scored = []
    for f in tqdm(pool):
        scored.append((f, get_sharpness(os.path.join(IMG_DIR, f))))
    scored.sort(key=lambda x: x[1]) # 按分数从小到大排

    # 执行你的切片要求
    # 1. Low: 指标最低的 1000 张
    low = scored[:1000]
    # 2. High: 指标最高的 1000 张
    high = scored[-1000:]
    # 3. Mixed: 前1000张的后500张 (500-1000) + 最后1000张的前500张 (-1000至-500)
    mixed = scored[500:1000] + scored[-1000:-500]

    return {"Low": low, "High": high, "Mixed": mixed}

# --------------------------------------------------------------------------------
# 3. 类别提取与统计
# --------------------------------------------------------------------------------
def count_classes(subset_dict):
    data_list = []
    for name, subset in subset_dict.items():
        print(f"Counting classes for {name} group...")
        counts = {c: 0 for c in CLASSES}
        for img_name, _ in tqdm(subset):
            # 获取对应的 mask 文件名
            mask_path = os.path.join(ANN_DIR, img_name.replace('.jpg', '_labelTrainIds.png'))
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                unique_ids = np.unique(mask)
                for uid in unique_ids:
                    if uid < len(CLASSES): # 过滤 255 等 ignore 标签
                        counts[CLASSES[uid]] += 1
        
        for cls, count in counts.items():
            data_list.append({"Experiment": name, "Class": cls, "Count": count})
    
    return pd.DataFrame(data_list)

# --------------------------------------------------------------------------------
# 4. 可视化
# --------------------------------------------------------------------------------
def plot_dist(df):
    # 为了清晰，我们只展示在全集中出现频率最高的前 30 个类
    top_classes = df.groupby('Class')['Count'].sum().nlargest(30).index
    plot_df = df[df['Class'].isin(top_classes)]

    plt.figure(figsize=(16, 8))
    sns.set_style("whitegrid")
    # 使用你喜欢的配色
    palette = {"Low": "#e74c3c", "High": "#3498db", "Mixed": "#2ecc71"}
    
    sns.barplot(data=plot_df, x='Class', y='Count', hue='Experiment', palette=palette)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.title("Class Distribution Comparison: Low vs High vs Mixed Sharpness", fontsize=16, fontweight='bold')
    plt.ylabel("Image-level Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig("stratified_class_distribution.png", dpi=300)
    print("Plot saved as stratified_class_distribution.png")

if __name__ == "__main__":
    subsets = analyze_subsets()
    df = count_classes(subsets)
    plot_dist(df)