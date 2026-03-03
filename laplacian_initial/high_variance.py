import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)['coco_stuff']

def visualize_gaps(root_path=".", top_n=20):
    # 1. 加载数据
    low = load_json(os.path.join(root_path, 'low.json'))
    high = load_json(os.path.join(root_path, 'high.json'))
    mixed = load_json(os.path.join(root_path, 'mixed.json'))

    # 2. 计算每个类别的表现跨度 (Gap)
    gap_data = []
    classes = list(low['per_class'].keys())
    
    for cls in classes:
        scores = {
            "Low": low['per_class'][cls]['IoU'],
            "High": high['per_class'][cls]['IoU'],
            "Mixed": mixed['per_class'][cls]['IoU']
        }
        max_val = max(scores.values())
        min_val = min(scores.values())
        gap = max_val - min_val
        
        # 记录数据
        gap_data.append({
            "Class": cls,
            "Low": scores["Low"],
            "High": scores["High"],
            "Mixed": scores["Mixed"],
            "Gap": gap,
            "Winner": max(scores, key=scores.get)
        })

    # 3. 筛选出 Gap 最大的前 N 个类别
    df_gap = pd.DataFrame(gap_data).sort_values(by='Gap', ascending=False).head(top_n)

    # 4. 绘图：Lollipop Chart
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="white")
    
    # 绘制水平线 (Gap 跨度)
    plt.hlines(y=df_gap['Class'], xmin=df_gap[['Low', 'High', 'Mixed']].min(axis=1), 
               xmax=df_gap[['Low', 'High', 'Mixed']].max(axis=1), color='grey', alpha=0.5, linewidth=2)
    
    # 绘制三个策略的点
    plt.scatter(df_gap['Low'], df_gap['Class'], color='#e74c3c', label='Low', s=100, zorder=3)
    plt.scatter(df_gap['High'], df_gap['Class'], color='#3498db', label='High', s=100, zorder=3)
    plt.scatter(df_gap['Mixed'], df_gap['Class'], color='#2ecc71', label='Mixed', s=100, zorder=3)

    # 修饰图表
    plt.title(f'Top {top_n} Categories with Largest Performance Gaps', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('mIoU (%)', fontsize=12)
    plt.ylabel('Category', fontsize=12)
    plt.legend(title='Sampling Strategy', loc='lower right', frameon=True)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    
    # 在行末标注 Gap 数值
    for i, row in df_gap.iterrows():
        plt.text(row[['Low', 'High', 'Mixed']].max() + 1, row['Class'], f"Δ={row['Gap']:.1f}", 
                 va='center', fontsize=10, fontweight='bold', color='#34495e')

    plt.tight_layout()
    plt.savefig('significant_performance_gaps.png', dpi=300)
    print(f"已生成敏感类别对比图: significant_performance_gaps.png")

if __name__ == "__main__":
    visualize_gaps()