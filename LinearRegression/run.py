"""
    线性变换
    v1.0: 初始版本--2025-11-13
        * main.py: 主函数入口
"""

import numpy as np
from plot_tookit import PlotToolkit

class Linear_Regression_Model:
    def __init__(self):
        # 初始化模型参数
        self.w = 0.0
        self.b = 0.0

        # 保存训练过程信息，用于绘图
        self.loss_history = []
        self.w_history = []
        self.b_history = []
        self.dw_history = []
        self.db_history = []

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

            # 计算并记录损失
            loss = np.mean((Y - Y_pred) ** 2)
            self.loss_history.append(loss)
            self.w_history.append(self.w)
            self.b_history.append(self.b)
            self.dw_history.append(dw)
            self.db_history.append(db)
            if epoch % log_interval == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, w: {self.w:.4f}, b: {self.b:.4f}")

        
def make_LinearData(Sample_num: int, Feature_dim : int = 1, Noise_level: float = 0.0, set_w: float = 4.57, set_b: float = 0.64):
    """
    生成线性回归数据集
    """
    X = np.random.rand(Sample_num, Feature_dim) * 10
    Y = set_w * X + set_b + Noise_level * np.random.randn(Sample_num, Feature_dim)
    return X, Y

# 主函数
def main():
    # 设置超参数
    hyperparams = {
        # 数据
        "set_w": 4.57,
        "set_b": 0.64,
        "sample_num": 100,
        "feature_dim": 1,
        "noise_level": 0.1,
        # 训练
        "lr": 1e-3,
        "epochs": 100,
        "log_interval": 5,
        "seed": 42
    }
        # 固定随机种子，确保结果可复现
    np.random.seed(42)
    # 生成数据集并训练模型
    set_w, set_b = hyperparams["set_w"], hyperparams["set_b"]
    sample_num, feature_dim = hyperparams["sample_num"], hyperparams["feature_dim"]
    X, Y = make_LinearData(sample_num, feature_dim, hyperparams["noise_level"], set_w=set_w, set_b=set_b)
    
    # 训练模型
    lr, epochs = hyperparams["lr"], hyperparams["epochs"]
    if feature_dim == 1:
        model = Linear_Regression_Model()
    else:         # 多变量线性回归模型
        print("Only single variable linear regression is supported.")
        return
    print(f"Trained parameters: w = {model.w}, b = {model.b}")

    # 画图
    plot_toolkit = PlotToolkit()
    plot_toolkit.plot_regression_results(X, Y, model, hyperparams)

if __name__ == "__main__":
    main()
