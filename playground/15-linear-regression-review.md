# 15 复盘线性回归

这一页复盘线性回归训练循环：参数为什么会更新、loss 为什么会下降、学习率为什么不能太大或太小。

## 最小训练循环

```python
import torch
X = torch.randn(1000, 2)
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2
y = X @ true_w + true_b + torch.randn(1000) * 0.01
w = torch.randn(2, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
for epoch in range(5):
    y_hat = X @ w + b
    loss = ((y_hat - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.03 * w.grad
        b -= 0.03 * b.grad
        w.grad.zero_()
        b.grad.zero_()
    print(epoch, loss.item())
```

## 每一行在做什么

- `X = torch.randn(1000, 2)`：创建 1000 条样本，每条样本有 2 个特征。
- `true_w = torch.tensor([2.0, -3.4])`：设置真实权重，用来合成标签。
- `true_b = 4.2`：设置真实偏置。
- `y = X @ true_w + true_b + noise`：生成标签，噪声模拟真实世界中的测量误差。
- `w = torch.randn(2, requires_grad=True)`：初始化要学习的权重，并让 PyTorch 跟踪梯度。
- `b = torch.zeros(1, requires_grad=True)`：初始化要学习的偏置。
- `for epoch in range(5)`：重复训练 5 轮。
- `y_hat = X @ w + b`：用当前参数做预测。
- `loss = ((y_hat - y) ** 2).mean()`：计算均方误差，衡量预测和真实标签的差距。
- `loss.backward()`：反向传播，计算 `loss` 对 `w` 和 `b` 的梯度。
- `with torch.no_grad()`：更新参数时不需要继续建立计算图。
- `w -= 0.03 * w.grad`：沿梯度反方向更新权重。
- `b -= 0.03 * b.grad`：沿梯度反方向更新偏置。
- `w.grad.zero_()` 和 `b.grad.zero_()`：清空梯度，避免下一轮梯度累加。
- `print(epoch, loss.item())`：观察每轮 loss 是否下降。

## 参数为什么会更新

`loss.backward()` 会告诉我们：如果某个参数增加一点，loss 会怎样变化。这个变化方向就是梯度。梯度指向 loss 上升最快的方向，所以训练时使用：

`参数 = 参数 - 学习率 * 梯度`

这表示往梯度的反方向走，希望 loss 变小。

## loss 为什么会下降

如果学习率合适，参数每次都会朝着让预测更接近标签的方向移动。预测误差变小，均方误差就会下降。

## 学习率为什么不能太大或太小

- 学习率太小：每一步走得太慢，loss 降得很慢，训练时间变长。
- 学习率太大：每一步跨得太远，可能越过最低点，导致 loss 震荡甚至变大。
- 合适学习率：loss 大体稳定下降，参数逐渐接近真实值。

## 我现在应该记住什么

线性回归训练循环的核心是：前向预测 -> 计算损失 -> 反向传播 -> 更新参数 -> 清空梯度。后面更复杂的神经网络，本质上也是这条闭环。
