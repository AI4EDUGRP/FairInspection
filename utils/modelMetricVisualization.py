import pandas as pd 
import numpy as np
from sklearn import metrics
from pyecharts.charts import *
import pyecharts.options as opts


def getMetrics(modelResult):
    resultDf = pd.read_csv(modelResult)
    fpr, tpr, _ = metrics.roc_curve(resultDf.y_label, resultDf.y_pred_score.values)
    cmatrix =  metrics.confusion_matrix(resultDf.y_label,resultDf.pred_label)
    fpr =[np.round(val, 4) for val in fpr]
    tpr=[np.round(val,4) for val in tpr]
    acc = np.round(metrics.accuracy_score(resultDf.y_label,resultDf.pred_label),3)
    prec = np.round(metrics.precision_score(resultDf.y_label,resultDf.pred_label),3)
    recall = np.round(metrics.recall_score(resultDf.y_label,resultDf.pred_label),3)
    f1score = np.round(metrics.f1_score(resultDf.y_label,resultDf.pred_label),3)
    return fpr, tpr, cmatrix, acc, prec, recall, f1score


def getBarMetrics(acc,prec,recall,f1score):
    bar=(
        Bar(init_opts=opts.InitOpts(width='300px',height='400px'))
        .add_xaxis(['Accuracy','Precision','Recall','F1-Score'])
        .add_yaxis(
            '',
            [acc,prec,recall,f1score],
            label_opts=opts.LabelOpts(is_show=False)
            )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_='category'
            )
        )
    )
    return bar


def confusionMatrix(cmatrix):
    value =[[0,1,int(cmatrix[0][0])],[0,0,int(cmatrix[0][1])],[1,1,int(cmatrix[1][0])],[1,0,int(cmatrix[1][1])]]
    heatmap = (
        HeatMap(init_opts=opts.InitOpts(width='300px',height='400px'))
        .add_xaxis([0,1])
        .add_yaxis(
            'Confusion Matrix',
            [1,0],
            value,
            label_opts=opts.LabelOpts(is_show=True,position='inside')
            )
        .set_global_opts(
            # title_opts=opts.TitleOpts(title="Confusion Matrix"),
            visualmap_opts=opts.VisualMapOpts(is_show=False),
            legend_opts=opts.LegendOpts(is_show=False),
            xaxis_opts=opts.AxisOpts(
                type_="category",
                name='True',
                name_location ='center',
                name_gap=15,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            yaxis_opts=opts.AxisOpts(
                type_="category",
                name='Predict',
                name_location ='center',
                name_gap=15,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
        )
    )
    return heatmap


def plotROC(fpr,tpr):
    roc = (
        Line(init_opts=opts.InitOpts(width='300px',height='400px'))
        .add_xaxis(fpr)
        .add_yaxis(
            'ROC Curve',
            tpr,
            label_opts=opts.LabelOpts(is_show=False),
            symbol='none',
            is_smooth=True)
        .add_yaxis(
            'Random Probabilities',
            fpr,
            label_opts=opts.LabelOpts(is_show=False),
            symbol='none',
            )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_='value',
                name='FPR',
                name_location='center',
                name_gap=25
            ),
            yaxis_opts=opts.AxisOpts(
                type_='value',
                name='TPR',
                name_location='center',
                name_gap=15
            ),
            tooltip_opts=opts.TooltipOpts(trigger='axis')
        )
    )
    return roc



