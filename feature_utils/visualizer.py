import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm

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

def count_classes_in_subset(path_list, ann_dir):
    """
    统计子集中每个类别出现的图片张数 (Image-level frequency)
    """
    class_counts = {c: 0 for c in CLASSES}
    
    print(f"Counting class occurrences in {len(path_list)} masks...")
    for img_path in tqdm(path_list):
        # 转换文件名: 2017_000000123456.jpg -> 2017_000000123456_labelTrainIds.png
        mask_name = os.path.basename(img_path).replace('.jpg', '_labelTrainIds.png')
        mask_path = os.path.join(ann_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            
            unique_ids = np.unique(mask)
            for uid in unique_ids:
                if uid < len(CLASSES):
                    class_counts[CLASSES[uid]] += 1
                    
    return class_counts

def plot_metric_distribution(scores_dict, metric_name):
    scores = list(scores_dict.values())
    plt.figure(figsize=(10, 6))
    sns.histplot(scores, kde=True, color='purple')
    plt.title(f"Global Distribution of {metric_name} (N={len(scores)})")
    plt.xlabel("Score Value")
    plt.ylabel("Frequency")
    plt.savefig(f"plot_1_{metric_name}_distribution.png")
    plt.close()

def plot_class_distribution_comparison(path_lists, names, ann_dir, metrics_name="Metrics"):
    """
    names: ['Low', 'High', 'Mixed']
    """
    data = []
    for paths, name in zip(path_lists, names):
        counts = count_classes_in_subset(paths, ann_dir)
        for cls, count in counts.items():
            data.append({"Strategy": name, "Class": cls, "Count": count})
    
    df = pd.DataFrame(data)
    
    # 筛选出在所有策略中出现频率最高的前 30 个类，否则图表太挤
    top_classes = df.groupby('Class')['Count'].sum().nlargest(30).index
    df_plot = df[df['Class'].isin(top_classes)]
    
    plt.figure(figsize=(18, 8))
    sns.barplot(data=df_plot, x='Class', y='Count', hue='Strategy', palette='muted')
    plt.xticks(rotation=45, ha='right')
    plt.title("Plot 4: Class Distribution Alignment Check (Top 30 Classes)")
    plt.tight_layout()
    plt.savefig(f"plot_4_class_alignment_{metrics_name}.png")
    plt.close()
    print(f"Saved plot_4_class_alignment_{metrics_name}.png")