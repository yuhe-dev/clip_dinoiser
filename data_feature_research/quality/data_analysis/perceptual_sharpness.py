import cv2
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import seaborn as sns
from scipy.stats import rankdata


def calculate_sharpness(img_path):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算拉普拉斯算子响应的方差
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score


SAMPLE_COUNT = 20000
img_dir = "../../../data/coco_stuff164k/images/train2017"

scores = []
all_img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.JPEG', '.png'))]
random.seed(42)
num_to_samples = min(SAMPLE_COUNT, len(all_img_names))
img_names = random.sample(all_img_names, num_to_samples)

for name in tqdm(img_names):
    path = os.path.join(img_dir, name)
    try:
        scores.append(calculate_sharpness(path))
    except:
        continue

# 1. 原始分数计算
scores = np.array(scores)

# 2. 对数转换 (Log Transformation)
# Laplacian Variance 跨度很大，log1p (ln(1+x)) 可以有效压缩动态范围并拉伸分布
log_scores = np.log1p(scores)

# 3. 百分位转换 (Percentile Ranking)
# 将分数转换为 0-1 之间的百分比排名，解决分布集中导致的视觉差异小的问题
percentile_scores = rankdata(scores) / len(scores)

# ------------------------------------------------------------------
# 修改后的可视化函数：支持对数或百分位
# ------------------------------------------------------------------
def plot_sharpness_distribution_v2(data, title, xlabel, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 标注 20% 和 80% 的位置
    p20 = np.percentile(data, 20)
    p80 = np.percentile(data, 80)
    plt.axvline(p20, color='red', linestyle='--', label='Bottom 20% (Blurry)')
    plt.axvline(p80, color='green', linestyle='--', label='Top 20% (Sharp)')
    
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

# 4. 调用：对比原始 vs 对数
# 你可以发现对数处理后的分布会更接近正态分布，更容易观察异常值
plot_sharpness_distribution_v2(log_scores, 
                               'Log-Transformed Sharpness Distribution', 
                               'Log(1 + Laplacian Variance)', 
                               'sharpness_log_dist.png')

# ------------------------------------------------------------------
# 修改后的画廊展示：使用百分位作为 Score 显示
# ------------------------------------------------------------------
def plot_sharpness_gallery_v2(img_names, original_scores, display_scores, img_dir, num_samples=5):
    # 根据原始分数排序
    sorted_idx = np.argsort(original_scores)
    
    indices_to_show = {
        "Blurry (Lowest 5)": sorted_idx[:num_samples],
        "Median (Middle 5)": sorted_idx[len(original_scores)//2 : len(original_scores)//2 + num_samples],
        "Sharp (Highest 5)": sorted_idx[-num_samples:]
    }
    
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 10))
    
    for row_idx, (title, idxs) in enumerate(indices_to_show.items()):
        for col_idx, idx in enumerate(idxs):
            img_path = os.path.join(img_dir, img_names[idx])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            # 这里显示百分位得分，例如 0.995 代表这张图比 99.5% 的图都清晰
            ax.set_title(f"Rank: {display_scores[idx]:.1%}\n(Raw: {int(original_scores[idx])})", fontsize=9)
            ax.axis('off')
            
            if col_idx == 0:
                ax.text(-0.2, 0.5, title, transform=ax.transAxes, rotation=90, 
                        va='center', ha='right', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("sharpness_gallery_percentile.png")

# 使用百分位进行画廊展示
plot_sharpness_gallery_v2(img_names, scores, percentile_scores, img_dir)