"""
    绘图工具包
    v1.0: 初始版本--2025-11-14

"""

import os
import numpy as np
from matplotlib import pyplot as plt

class PlotToolkit:
    def __init__(self, save_dir: str = "results/"):
        self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def plot_regression_results(self, X, Y, model, hyperparams: dict):
        """
        绘制线性回归结果
        """

        fig = plt.figure(figsize=(12, 10))

        # 子图1: 数据点与回归线
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.scatter(X, Y, color='blue', label='Data Points')
        x_line = np.linspace(np.min(X), np.max(X), 100)
        y_line = model.w * x_line + model.b
        ax1.plot(x_line, y_line, color='red', label='Regression Line')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Linear Regression Results')
        ax1.legend()

        # 子图2: 损失下降曲线
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.plot(range(len(model.loss_history)), model.loss_history, color='green', label='Loss')
        ax2.plot(range(len(model.dw_history)), model.dw_history, color='orange', label='Gradient of w')
        ax2.plot(range(len(model.db_history)), model.db_history, color='purple', label='Gradient of b')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.set_title('Loss Decrease Over Epochs')

        # 子图3: 参数变化曲线
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(range(len(model.w_history)), model.w_history, color='orange', label='w')
        ax3.plot(range(len(model.b_history)), model.b_history, color='purple', label='b')
        ax3.hlines(hyperparams["set_w"], 0, len(model.w_history), colors='red', linestyles='dashed', label='True w')
        ax3.hlines(hyperparams["set_b"], 0, len(model.b_history), colors='blue', linestyles='dashed', label='True b')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Parameter Values')
        ax3.set_title('Parameter Changes Over Epochs')
        ax3.legend()

        # 子图4： 训练超参数设置
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        textstr = f"w_predicted: {model.w:.4f}\nb_predicted: {model.b:.4f}\n"
        for k, v in hyperparams.items():
            textstr += f"{k}: {v}\n"
        ax4.text(0.1, 0.5, textstr, fontsize=12, verticalalignment='center')
        ax4.set_title('Training Parameters')

        plt.savefig(f"{self.save_dir}/regression_results.png")
        plt.close()

    def plot_regression_results_2D(self, X, Y, model, hyperparams: dict):
        """
        绘制线性回归结果
        """

        fig = plt.figure(figsize=(12, 10))

        # 子图1: 数据点与回归线
        # ax1 = fig.add_subplot(2, 2, 1)
        # ax1.scatter(X, Y, color='blue', label='Data Points')
        # x_line = np.linspace(np.min(X), np.max(X), 100)
        # y_line = model.w * x_line + model.b
        # ax1.plot(x_line, y_line, color='red', label='Regression Line')
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # ax1.set_title('Linear Regression Results')
        # ax1.legend()

        # 子图2: 损失下降曲线
        ax2 = fig.add_subplot(2, 2, 1)
        ax2.plot(range(len(model.loss_history)), model.loss_history, color='green', label='Loss')
        ax2.plot(range(len(model.dw_history)), model.dw_history[:,0], color='orange', label='Gradient of w0')
        ax2.plot(range(len(model.dw_history)), model.dw_history[:,1], color='red', label='Gradient of w1')
        ax2.plot(range(len(model.db_history)), model.db_history, color='purple', label='Gradient of b')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.set_title('Loss Decrease Over Epochs')

        # 子图3: 参数变化曲线
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.plot(range(len(model.w_history[:,0])), model.w_history[:,0], color='orange', label='w0')
        ax3.plot(range(len(model.w_history[:,1])), model.w_history[:,1], color='red', label='w1')
        ax3.plot(range(len(model.b_history)), model.b_history, color='purple', label='b')
        ax3.hlines(hyperparams["set_w"][0], 0, len(model.w_history), colors='red', linestyles='dashed', label='True w0')
        ax3.hlines(hyperparams["set_w"][1], 0, len(model.w_history), colors='red', linestyles='dashed', label='True w1')
        ax3.hlines(hyperparams["set_b"], 0, len(model.b_history), colors='blue', linestyles='dashed', label='True b')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Parameter Values')
        ax3.set_title('Parameter Changes Over Epochs')
        ax3.legend()

        # 子图4： 训练超参数设置
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        textstr = f"w_predicted: {model.w:.4f}\nb_predicted: {model.b:.4f}\n"
        for k, v in hyperparams.items():
            textstr += f"{k}: {v}\n"
        ax4.text(0.1, 0.5, textstr, fontsize=12, verticalalignment='center')
        ax4.set_title('Training Parameters')

        plt.savefig(f"{self.save_dir}/regression_results.png")
        plt.close()