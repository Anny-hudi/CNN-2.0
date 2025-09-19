import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取训练日志
log_file = 'logs/I5R5_OHLC/I5R5_OHLC_18-20.csv'
df = pd.read_csv(log_file, index_col=0)

# 创建图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

epochs = range(1, len(df.columns) + 1)

# 训练和验证损失
ax1.plot(epochs, df.loc['train_loss'], 'b-', label='训练损失', linewidth=2)
ax1.plot(epochs, df.loc['valid_loss'], 'r-', label='验证损失', linewidth=2)
ax1.set_xlabel('轮次 (Epoch)')
ax1.set_ylabel('损失 (Loss)')
ax1.set_title('训练和验证损失曲线')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 训练和验证准确率
ax2.plot(epochs, df.loc['train_acc'], 'b-', label='训练准确率', linewidth=2)
ax2.plot(epochs, df.loc['valid_acc'], 'r-', label='验证准确率', linewidth=2)
ax2.set_xlabel('轮次 (Epoch)')
ax2.set_ylabel('准确率 (Accuracy)')
ax2.set_title('训练和验证准确率曲线')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 损失差异分析
loss_diff = np.array(df.loc['valid_loss']) - np.array(df.loc['train_loss'])
ax3.plot(epochs, loss_diff, 'g-', linewidth=2)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax3.set_xlabel('轮次 (Epoch)')
ax3.set_ylabel('验证损失 - 训练损失')
ax3.set_title('过拟合分析 (验证损失 - 训练损失)')
ax3.grid(True, alpha=0.3)

# 准确率差异分析
acc_diff = np.array(df.loc['valid_acc']) - np.array(df.loc['train_acc'])
ax4.plot(epochs, acc_diff, 'purple', linewidth=2)
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax4.set_xlabel('轮次 (Epoch)')
ax4.set_ylabel('验证准确率 - 训练准确率')
ax4.set_title('泛化能力分析 (验证准确率 - 训练准确率)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印关键统计信息
print("=== 训练分析报告 ===")
print(f"最佳验证损失: {min(df.loc['valid_loss']):.6f} (第{df.loc['valid_loss'].idxmin()}轮)")
print(f"最佳验证准确率: {max(df.loc['valid_acc']):.4f} (第{df.loc['valid_acc'].idxmax()}轮)")
print(f"最终训练损失: {df.loc['train_loss'].iloc[-1]:.6f}")
print(f"最终验证损失: {df.loc['valid_loss'].iloc[-1]:.6f}")
print(f"最终训练准确率: {df.loc['train_acc'].iloc[-1]:.4f}")
print(f"最终验证准确率: {df.loc['valid_acc'].iloc[-1]:.4f}")

# 过拟合分析
final_loss_diff = df.loc['valid_loss'].iloc[-1] - df.loc['train_loss'].iloc[-1]
final_acc_diff = df.loc['valid_acc'].iloc[-1] - df.loc['train_acc'].iloc[-1]

print(f"\n=== 过拟合分析 ===")
print(f"最终损失差异: {final_loss_diff:.6f}")
print(f"最终准确率差异: {final_acc_diff:.4f}")

if final_loss_diff > 0.1:
    print("⚠️  存在明显过拟合现象")
elif final_loss_diff < -0.05:
    print("✅ 模型泛化能力良好")
else:
    print("✅ 过拟合程度适中")

if final_acc_diff < -0.05:
    print("⚠️  训练准确率明显高于验证准确率")
elif final_acc_diff > 0.05:
    print("✅ 验证准确率高于训练准确率，泛化能力好")
else:
    print("✅ 训练和验证准确率接近")
