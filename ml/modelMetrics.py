import pandas as pd 
from sklearn import metrics


def getMetrics(modelResult):
    resultDf = pd.read_csv(modelResult)
    fpr, tpr, _ = metrics.roc_curve(resultDf.y_label, resultDf.y_pred_score.values.to_list())
    return fpr, tpr

# data = 'data/lrModel/studDf.csv'
# tpr, tpr = getMetrics(data)
