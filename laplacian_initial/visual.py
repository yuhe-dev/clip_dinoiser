import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path, strategy_name):
    """读取并解析实验结果 JSON"""
    if not os.path.exists(file_path):
        print(f"警告: 文件 {file_path} 不存在，跳过。")
        return None, None
    
    with open(file_path, 'r') as f:
        content = json.load(f)['coco_stuff']
        
    # 提取 Summary
    summary = content['summary']
    summary['Strategy'] = strategy_name
    
    # 提取 Per Class
    per_class = []
    for cls_name, metrics in content['per_class'].items():
        per_class.append({
            "Class": cls_name,
            "Strategy": strategy_name,
            "IoU": metrics['IoU'],
            "Acc": metrics['Acc']
        })
    return summary, per_class

def generate_comparison_plots():
    # 1. 数据准备
    files = {
        "new_low.json": "Low",
        "new_mixed.json": "Mixed",
        "new_high.json": "High"
    }
    
    summaries = []
    all_class_data = []
    
    for filename, strategy in files.items():
        s, p = load_data(filename, strategy)
        if s:
            summaries.append(s)
            all_class_data.extend(p)
            
    df_summary = pd.DataFrame(summaries)
    df_class = pd.DataFrame(all_class_data)

    # ---------------------------------------------------------
    # 绘图 1: 全局指标对比 (mIoU, mAcc, aAcc)
    # ---------------------------------------------------------
    df_summary_melted = df_summary.melt(id_vars='Strategy', value_vars=['mIoU', 'mAcc', 'aAcc'])
    
    plt.figure(figsize=(11, 6)) # 稍微加宽一点画布
    ax = sns.barplot(data=df_summary_melted, x='variable', y='value', hue='Strategy', 
                     palette=['#E74C3C', '#2ECC71', '#3498DB'])
    
    plt.title('Global Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylim(df_summary_melted['value'].min() - 2, df_summary_melted['value'].max() + 3)
    
    # 核心修改：放在右侧框外
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Strategy", frameon=True)
    
    # 保存时必须加 bbox_inches='tight'，否则右侧图例会被切掉
    
    # 如果你更喜欢放在右侧，可以使用下面这行替换上面那行：
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # 在柱状图上标注数值
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)
    
    plt.savefig('comparison_global_right.png', dpi=300, bbox_inches='tight')
    print("已生成: comparison_global_right.png")

    # ---------------------------------------------------------
    # 绘图 2: 表现差异最大的 Top 20 类别 (Gap Analysis)
    # ---------------------------------------------------------
    # 计算每个类别的极差 (Max - Min)
    gap_df = df_class.groupby('Class')['IoU'].agg(lambda x: x.max() - x.min()).reset_index()
    top_gap_classes = gap_df.sort_values(by='IoU', ascending=False).head(20)['Class']
    df_top_gaps = df_class[df_class['Class'].isin(top_classes := top_gap_classes)]

    plt.figure(figsize=(12, 10))
    # 绘制 Lollipop Chart
    # 排序：按 Mixed 性能排序
    order = df_top_gaps[df_top_gaps['Strategy'] == 'Mixed'].sort_values('IoU')['Class']
    
    sns.pointplot(data=df_top_gaps, y='Class', x='IoU', hue='Strategy', order=order,
                  join=False, palette=['#E74C3C', '#2ECC71', '#3498DB'], markers=["o", "s", "D"], scale=0.8)
    
    # 画辅助线
    for i, cls in enumerate(order):
        row = df_top_gaps[df_top_gaps['Class'] == cls]
        plt.hlines(y=i, xmin=row['IoU'].min(), xmax=row['IoU'].max(), color='gray', alpha=0.3)

    plt.title('Top 20 Categories with Largest Performance Gap', fontsize=16, fontweight='bold')
    plt.xlabel('IoU (%)')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.legend(title='Strategy', bbox_to_anchor=(1, 0.15))
    plt.tight_layout()
    plt.savefig('comparison_class_gaps.png', dpi=300)
    print("已生成: comparison_class_gaps.png")

    # ---------------------------------------------------------
    # 绘图 3: 代表性物理特征组别对比
    # ---------------------------------------------------------
    representative_groups = {
        "Rigid Structure": ["bicycle", "car", "motorcycle", "train", "bus"],
        "Organic/Texture": ["broccoli", "pizza", "cake", "banana", "person"],
        "Background/Stuff": ["grass", "tree", "sky-other", "clouds", "mountain"],
        "Small/Fine": ["traffic light", "bottle", "cup", "knife", "scissors"]
    }
    
    plot_rows = []
    for group_name, classes in representative_groups.items():
        for cls in classes:
            item = df_class[df_class['Class'] == cls].copy()
            item['Group'] = group_name
            plot_rows.append(item)
    
    df_rep = pd.concat(plot_rows)

    g = sns.catplot(data=df_rep, kind="bar", x="IoU", y="Class", hue="Strategy", 
                    col="Group", col_wrap=2, palette=['#E74C3C', '#2ECC71', '#3498DB'],
                    height=5, aspect=1.2, sharey=False)
    
    g.fig.suptitle('Performance Insight by Representative Groups', fontsize=18, fontweight='bold', y=1.05)
    plt.savefig('comparison_representative_groups.png', dpi=300, bbox_inches='tight')
    print("已生成: comparison_representative_groups.png")

if __name__ == "__main__":
    generate_comparison_plots()