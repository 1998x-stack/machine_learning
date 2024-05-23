# 定义AdaGrad, RMSProp和Adam类，并实现其优化方法
from typing import List, Callable, Tuple
import numpy as np
"""
1. **AdaGrad**:
    - 初始化：设置学习率 `alpha` 和小数值 `epsilon` 防止除以零。初始化梯度平方和为零。
    - 更新规则：累加梯度的平方和，调整梯度，更新参数。
2. **RMSProp**:
    - 初始化：设置学习率 `alpha`、衰减率 `beta` 和小数值 `epsilon`。初始化梯度平方的移动平均为零。
    - 更新规则：计算梯度平方的移动平均，调整梯度，更新参数。
3. **Adam**:
    - 初始化：设置学习率 `alpha`、一阶矩衰减率 `beta1`、二阶矩衰减率 `beta2` 和小数值 `epsilon`。初始化一阶和二阶矩为零，时间步长为零。
    - 更新规则：更新一阶和二阶矩，计算偏差校正，调整梯度，更新参数。
"""
class AdaGrad:
    def __init__(self, alpha: float = 0.01, epsilon: float = 1e-8):
        self.alpha = alpha
        self.epsilon = epsilon
        self.g_sum = None

    def update(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.g_sum is None:
            self.g_sum = np.zeros_like(grad)
        self.g_sum += grad ** 2
        adjusted_grad = grad / (np.sqrt(self.g_sum) + self.epsilon)
        theta -= self.alpha * adjusted_grad
        return theta


class RMSProp:
    def __init__(self, alpha: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.v = None

    def update(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.v is None:
            self.v = np.zeros_like(grad)
        self.v = self.beta * self.v + (1 - self.beta) * grad ** 2
        adjusted_grad = grad / (np.sqrt(self.v) + self.epsilon)
        theta -= self.alpha * adjusted_grad
        return theta


class Adam:
    def __init__(self, alpha: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, theta: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.m is None:
            self.m = np.zeros_like(grad)
        if self.v is None:
            self.v = np.zeros_like(grad)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        theta -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return theta


# 测试这些优化器类
def test_optimizer(optimizer_class, grad_fn: Callable[[np.ndarray], np.ndarray], initial_theta: np.ndarray, iterations: int = 100):
    optimizer = optimizer_class()
    theta = initial_theta
    for i in range(iterations):
        grad = grad_fn(theta)
        theta = optimizer.update(theta, grad)
        print(f"Iteration {i+1}: theta = {theta}")

# 示例梯度函数：简单的二次函数的梯度
def simple_quadratic_grad(theta: np.ndarray) -> np.ndarray:
    return 2 * theta

initial_theta = np.array([5.0, 5.0])

# 测试AdaGrad
print("Testing AdaGrad:")
test_optimizer(AdaGrad, simple_quadratic_grad, initial_theta)

# 测试RMSProp
print("\nTesting RMSProp:")
test_optimizer(RMSProp, simple_quadratic_grad, initial_theta)

# 测试Adam
print("\nTesting Adam:")
test_optimizer(Adam, simple_quadratic_grad, initial_theta)