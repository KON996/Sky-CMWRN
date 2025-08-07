import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

def calculate_metrics(pred, target):
    """
 计算各种评估指标
 Args:
     pred: 预测值（需为 numpy 数组）
     target: 真实值（需为 numpy 数组）
 Returns:
     rmse: Root Mean Square Error
     corr: Pearson correlation coefficient
     sagr: Sign Agreement（符号一致性指标）
     ccc: Concordance Correlation Coefficient
 """
    # RMSE指标
    rmse = np.sqrt(mean_squared_error(target, pred))

    # corr指标
    corr, _ = pearsonr(pred, target)

    # Sign Agreement (SAGR) 符号一致性指标
    sign_match = np.sign(pred) == np.sign(target)
    sagr = np.mean(sign_match)

    # CCC指标
    mean_pred = np.mean(pred)
    mean_target = np.mean(target)
    var_pred = np.var(pred, ddof=1)
    var_target = np.var(target, ddof=1)

    covariance = np.cov(pred, target)[0,1]
    ccc = (2 * covariance) / (var_pred + var_target + (mean_pred - mean_target)**2)

    return rmse, corr, sagr, ccc

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count