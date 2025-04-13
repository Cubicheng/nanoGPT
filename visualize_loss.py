import matplotlib.pyplot as plt

# 读取数据
epochs = []
losses = []

with open("log/log(loss).txt", "r") as file:  # 替换为您的文件名
    for line in file:
        parts = line.strip().split()
        epochs.append(int(parts[0]))
        losses.append(float(parts[-1]))

# 绘制曲线
plt.figure(figsize=(10, 6))
plt.plot(epochs, losses, "b-", linewidth=2, label="Training Loss")
plt.title("Training Loss Curve", fontsize=16)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12)

# 自动调整刻度标签
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 显示图表
plt.tight_layout()
plt.show()
