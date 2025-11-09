import matplotlib.pyplot as plt
import numpy as np

# 数据
models = {
    "with_positional_encoding": {
        "loss": 2.7530321694012447,
        "accuracy": 0.5412312524527142,
        "perplexity": 15.690134976257978
    },
    "without_positional_encoding": {
        "loss": 2.772958646336184,
        "accuracy": 0.5404035297349149,
        "perplexity": 16.00591988042815
    }
}

plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建三个子图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 提取数据
with_pe = models["with_positional_encoding"]
without_pe = models["without_positional_encoding"]

# 1. 最终损失对比
metrics_loss = ['With PE', 'Without PE']
values_loss = [with_pe['loss'], without_pe['loss']]
colors_loss = ['lightblue', 'lightcoral']

bars1 = axes[0].bar(metrics_loss, values_loss, color=colors_loss, alpha=0.7, edgecolor='black')
axes[0].set_title('Loss Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# 在柱子上添加数值标签
for bar, value in zip(bars1, values_loss):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# 2. 最终准确率对比
values_accuracy = [with_pe['accuracy'], without_pe['accuracy']]
colors_accuracy = ['lightgreen', 'lightyellow']

bars2 = axes[1].bar(metrics_loss, values_accuracy, color=colors_accuracy, alpha=0.7, edgecolor='black')
axes[1].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

# 在柱子上添加数值标签
for bar, value in zip(bars2, values_accuracy):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# 3. 最终困惑度对比
values_perplexity = [with_pe['perplexity'], without_pe['perplexity']]
colors_perplexity = ['lightpink', 'lightgray']

bars3 = axes[2].bar(metrics_loss, values_perplexity, color=colors_perplexity, alpha=0.7, edgecolor='black')
axes[2].set_title('Perplexity Comparison', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Perplexity', fontsize=12)
axes[2].grid(axis='y', alpha=0.3)

# 在柱子上添加数值标签
for bar, value in zip(bars3, values_perplexity):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('model_comparison_metrics.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()

