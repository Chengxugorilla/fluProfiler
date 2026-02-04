"""
Performance metrics for fluProfiler evaluation.
"""

import numpy as np
import torch
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


def print_exams(Observation, Prediction, print_result=True):
    O = Observation
    P = Prediction
    
    MAE, MSE, Pearson, Spearman, R2 = mean_absolute_error(O, P), mean_squared_error(O, P), pearsonr(O, P), spearmanr(O, P), r2_score(O, P)
    if print_result:
        print(f"MAE: {MAE:.5f}\nMSE: {MSE:.5f}\npearson correlation: {Pearson[0]:.5f}\nspearman correlation: {Spearman[0]:.5f}\nR2_score: {R2:.5f}")

    return MAE, MSE, Pearson, Spearman, R2

class EarlyStopping:
    def __init__(self, patience=7, verbose=True, delta=0, trace_func=print, save_dir=None):
        """
        初始化早停类

        参数:
            patience (int): 在早停之前允许性能没有改善的epochs数量
            verbose (bool): 如果为True，则打印一条消息，每次更新时
            delta (float): 最小变化阈值以认定为改善
            path (str): 保存模型的文件路径
            trace_func (function): 用于输出消息的函数
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.inf
        self.early_stop = False
        self.delta = delta
        self.save_dir = save_dir
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        """
        调用早停逻辑检查

        参数:
            val_loss (float): 当前验证集的MSE loss
            model (torch.nn.Module): 需要保存的PyTorch模型
        """
        if self.best_score > val_loss:
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_score = val_loss
        else:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """
        保存模型当验证集召回率提升时

        参数:
            val_recall (float): 当前验证集的召回率
            model (torch.nn.Module): 需要保存的PyTorch模型
        """
        if self.verbose:
            self.trace_func(
                f'Validation MSE decrease ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')

        current_time = datetime.now()

        # 格式化当前时间为字符串，例如：2024-06-09_12-34-56
        checkpoint_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(model, self.save_dir + checkpoint_name + '.pth')
