"""
    线性变换
    v1.0: 初始版本--2025-11-13
        * main.py: 主函数入口
"""

import numpy as np

class Linear_Regression_Model:
    def __init__(self):
        # 初始化模型参数
        self.w = 0.0
        self.b = 0.0

    def predict(self, X):
        # 预测函数
        return self.w * X + self.b
    
    def fit(self, X, Y, lr = 1e-3, epochs = 1000, log_interval = 100):
        # 训练函数，使用梯度下降法
        print("Starting training...")
        print(f"Initial parameters: w = {self.w}, b = {self.b}")
        print(f"Learning rate: {lr}, Epochs: {epochs}")
        print(f"Initial Loss: {np.mean((Y - self.predict(X)) ** 2):.4f}")
        for epoch in range(epochs):
            Y_pred = self.predict(X)
            # 计算梯度 MSE
            dw = -2 * np.mean(X * (Y - Y_pred))
            db = -2 * np.mean(Y - Y_pred)
            # 更新参数
            self.w -= lr * dw
            self.b -= lr * db
            if epoch % log_interval == 0 or epoch == epochs - 1:
                loss = np.mean((Y - Y_pred) ** 2)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, w: {self.w:.4f}, b: {self.b:.4f}")

        
def make_LinearData(Sample_num: int, Feature_dim : int = 1, Noise_level: float = 0.0):
    """
    生成线性回归数据集
    """
    X = np.random.rand(Sample_num, Feature_dim) * 10
    Y = 4.57 * X + 0.64 + Noise_level * np.random.randn(Sample_num, Feature_dim)
    return X, Y

# 主函数
def main():
    # 固定随机种子，确保结果可复现
    np.random.seed(42)
    X, Y = make_LinearData(100, 1, 0.1)
    model = Linear_Regression_Model()
    model.fit(X, Y, lr=1e-6, epochs=500000, log_interval=5000)
    print(f"Trained parameters: w = {model.w}, b = {model.b}")

if __name__ == "__main__":
    main()
