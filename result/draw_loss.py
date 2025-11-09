import matplotlib.pyplot as plt
import numpy as np

# 两个模型的训练损失数据
train_losses_1 =[
    5.281054948957429,
    4.1671841435509345,
    3.7212142709073093,
    3.4606923890131207,
    3.289750964492359,
    3.171071823077213,
    3.0846386262005208,
    3.032968506741248,
    2.995864255244258,
    2.9547296939070544
  ]

train_losses_2 =  [
    5.286145661414834,
    4.157430445191923,
    3.7158020614779246,
    3.463535400833858,
    3.296077818796314,
    3.178371164526674,
    3.0918625368122745,
    3.0401342342581237,
    3.0009952026783497,
    2.961461186191013
  ]

# 创建图表
plt.figure(figsize=(12, 8))

# 绘制两条折线
epochs = range(1, len(train_losses_1) + 1)
plt.plot(epochs, train_losses_1, linewidth=2, markersize=6, label='With PE')
plt.plot(epochs, train_losses_2, linewidth=2, markersize=6, label='Without PE')

# 设置图表标题和标签
plt.title('Training Loss Comparison Between Two Models', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Training Loss', fontsize=14)

# 设置坐标轴
plt.xlim(0.5, len(train_losses_1) + 0.5)
plt.xticks(epochs)
plt.grid(True, alpha=0.3)

# 添加图例
plt.legend(fontsize=12)

# 添加数值标注（可选，在最后一个点标注最终损失值）
plt.annotate(f'Final: {train_losses_1[-1]:.3f}',
             xy=(epochs[-1], train_losses_1[-1]),
             xytext=(10, 10), textcoords='offset points',
             fontsize=10, color='blue')

plt.annotate(f'Final: {train_losses_2[-1]:.3f}',
             xy=(epochs[-1], train_losses_2[-1]),
             xytext=(10, -15), textcoords='offset points',
             fontsize=10, color='red')

# 显示图表
plt.tight_layout()
plt.show()
