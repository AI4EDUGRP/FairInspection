from fairness.fairnessEvaluation import *
from pyecharts.charts import *
import pyecharts.options as opts
from pyecharts.options.global_options import ThemeType
import pandas as pd
import os


def getFairRslt(metric_test):
    fairMetric= ["Statistical Parity Difference", "Equal Opportunity Difference","Average odds Difference","Disparate Impact"]
    fairRslt=[]
    for metric in fairMetric:
        fairRslt.append(getFairnessMetric(metric_test,metric))
    return fairRslt

def getFairGauge(fairRslt,fmin,fmax,flow=0.45,fhigh=0.55):
    gauge = (
        Gauge(init_opts = opts.InitOpts(
                width='340px',
                height ='300px'
        ))
        .add(
            "",
            [('',fairRslt)],
            min_=fmin,
            max_=fmax,
            radius="100%",
            axisline_opts=opts.AxisLineOpts(
                linestyle_opts=opts.LineStyleOpts(
                    color=[(flow,'#fd666d'),(fhigh,"#37a2da"),(fmax,'#fd666d')],
                    width=25
                )
            ),
            detail_label_opts=opts.GaugeDetailOpts(
                formatter="{value}",
                # padding=[200,0,0,0],
                offset_center=[0,60]
            )
        )
        .set_global_opts(
            title_opts = opts.TitleOpts(),
            legend_opts = opts.LegendOpts(is_show=False)
        )
    )
    return gauge


def getThresholdFair(thresholdFairDf,metric,xlabel,ylabel):
    
    thresh = list(thresholdFairDf['thresh'].values)
    metricVal = [np.round(val,3) for val in thresholdFairDf[metric].values]
    scatter = (
        Scatter(init_opts = opts.InitOpts(
                width='340px',
                height ='300px'
        ))
        .add_xaxis(xaxis_data=thresh)
        .add_yaxis(
            series_name=ylabel,
            y_axis=metricVal,
            symbol_size=3,
            label_opts =opts.LabelOpts(is_show=False)
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_='value',
                name=xlabel,
                name_location='center',
                name_gap=25,
                splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            yaxis_opts=opts.AxisOpts(
                type_='value',
                # name=ylabel,
                # name_location='end',
                name_gap=25,
                axistick_opts=opts.AxisTickOpts(is_show=True),
                splitline_opts=opts.SplitLineOpts(is_show=True)
            ),
            tooltip_opts=opts.TooltipOpts(trigger='axis')
        )
    )
    return scatter