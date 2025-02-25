from pyecharts.charts import *
import pyecharts.options as opts
from pyecharts.commons.utils import JsCode

def StudentTrueScatter(studDfTruePass, studDfTrueFail):

    perfTrue=(
        Scatter(init_opts=opts.InitOpts(width='600px',height='400px'))
        .add_xaxis(studDfTruePass.iloc[:,1])
        .add_yaxis(
            'Pass',
            [list(z) for z in zip(studDfTruePass.iloc[:,2].tolist(),studDfTruePass.iloc[:,0].tolist())],
            label_opts=opts.LabelOpts(is_show=False),
            symbol_size=5
            )
        .add_yaxis(
            'Fail',
            [list(z) for z in zip(studDfTrueFail.iloc[:,2].tolist(),studDfTrueFail.iloc[:,0].tolist())],
            label_opts=opts.LabelOpts(is_show=False),
            symbol_size=5
            )
        .set_global_opts(
            yaxis_opts=opts.AxisOpts(
                axisline_opts=opts.AxisLineOpts(
                    is_on_zero=False
                )
            ),
            tooltip_opts=opts.TooltipOpts(
                formatter=JsCode(
                    """function(params){return 'ID'+params.value[2]+' : '+params.value[1];}"""
                )
            ),
            legend_opts=opts.LegendOpts(pos_left='5%')
        )
        .set_series_opts(
            itemstyle_opts=opts.ItemStyleOpts(
                border_color='black'
            )
        )
    )
    return perfTrue

def StudentPredScatter(studDfPredPass, studDfPredFail, studDfFNS, studDfFPS):
    # vals=0
    perfPred=(
        Scatter(init_opts=opts.InitOpts(width='600px',height='400px'))
        .add_xaxis(studDfPredPass.iloc[:,1])
        .add_yaxis(
            'Pass',
            [list(z) for z in zip(studDfPredPass.iloc[:,2].tolist(),studDfPredPass.iloc[:,0].tolist())],
            label_opts=opts.LabelOpts(is_show=False),
            symbol_size=5
            )
        .add_yaxis(
            'Fail',
            [list(z) for z in zip(studDfPredFail.iloc[:,2].tolist(),studDfPredFail.iloc[:,0].tolist())],
            label_opts=opts.LabelOpts(is_show=False),
            symbol_size=5
            )
        .add_yaxis(
            'False Negative Samples',
            [list(z) for z in zip(studDfFNS.iloc[:,2].tolist(),studDfFNS.iloc[:,0].tolist())],
            label_opts=opts.LabelOpts(is_show=False),
            symbol='rect',
            symbol_size=5
            )
        .add_yaxis(
            'False Positive Samples',
            [list(z) for z in zip(studDfFPS.iloc[:,2].tolist(),studDfFPS.iloc[:,0].tolist())],
            label_opts=opts.LabelOpts(is_show=False),
            symbol='triangle',
            symbol_size=5
            )
        
        .set_global_opts(
            yaxis_opts=opts.AxisOpts(
                axisline_opts=opts.AxisLineOpts(
                    is_on_zero=False
                )
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger_on="click",
                formatter=JsCode(
                    """function(params){
                        document.getElementById('studIdDiv').value=params.value[2];
                        return 'ID'+params.value[2]+' : '+params.value[1];}"""
                )
            ),
            legend_opts=opts.LegendOpts(pos_right='10%')
        )
        .set_series_opts(
            itemstyle_opts=opts.ItemStyleOpts(
                border_color='black'
            )
        )
    )
    return perfPred