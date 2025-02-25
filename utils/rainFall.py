from turtle import color
from pyecharts import options as opts
from pyecharts.charts import *
from pyecharts.options.global_options import ThemeType
from pyecharts.commons.utils import JsCode
from pyecharts.globals import CurrentConfig,NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_NOTEBOOK
from pyecharts.globals import ThemeType


def performance_rainfall(bins,male_performance,female_performance):
    l1 = (
        Line()
        .add_xaxis(xaxis_data=bins)
        .add_yaxis(
            series_name="Number of Male Students:",
            y_axis=male_performance,
            symbol_size=8,
            is_hover_animation=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(
                width=1.5,
            ),
            is_smooth=True,
            is_symbol_show=False
        )
        .set_global_opts(
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            axispointer_opts=opts.AxisPointerOpts(
                is_show=True,
                link=[{"xAxisIndex": "all"}]
            ),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=True,
                    is_realtime=True,
                    start_value=30,
                    end_value=70,
                    xaxis_index=[0, 1],
                    # pos_top='77%'
                )
            ],
            xaxis_opts=opts.AxisOpts(
                type_="category",
                boundary_gap=False,
                axisline_opts=opts.AxisLineOpts(is_on_zero=True),
            ),
            yaxis_opts=opts.AxisOpts(max_=max(male_performance), name_gap=23, name="Number of Students"),
            legend_opts=opts.LegendOpts(pos_left="0",pos_top='top'),
        )
    )

    l2 = (
        Line()
        .add_xaxis(xaxis_data=bins)
        .add_yaxis(
            series_name="Number of Female Students:",
            y_axis=female_performance,
            xaxis_index=1,
            yaxis_index=1,
            symbol_size=8,
            is_hover_animation=True,
            label_opts=opts.LabelOpts(is_show=False),
            linestyle_opts=opts.LineStyleOpts(
                width=1.5,
                color='red'
                ),
            is_smooth=True,
            is_symbol_show=False,
        )
        .set_global_opts(
            axispointer_opts=opts.AxisPointerOpts(
                is_show=True, 
                link=[{"xAxisIndex": "all"}]
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            xaxis_opts=opts.AxisOpts(
                grid_index=1,
                type_="category",
                boundary_gap=False,
                axisline_opts=opts.AxisLineOpts(is_on_zero=True),
                position="top",
            ),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_realtime=True,
                    type_="inside",
                    start_value=0,
                    end_value=1,
                    xaxis_index=[0, 1],
                )
            ],
            yaxis_opts=opts.AxisOpts(is_inverse=True, position='right',name_gap=20,name="Number of Students"),
            legend_opts=opts.LegendOpts(pos_top='top',pos_left="30%",textstyle_opts=opts.TextStyleOpts(color='red'),)
        )
    )

    rainfall=(
        Grid(init_opts=opts.InitOpts(width="1140px", height="500px",theme=ThemeType.LIGHT))
        .add(chart=l1,
             grid_opts=opts.GridOpts(pos_left=60, pos_right=60, pos_top="15%"),
            )
        .add(
            chart=l2,
            grid_opts=opts.GridOpts(pos_left=60, pos_right=60, pos_top="15%"),
        )
    )
    return rainfall

def getGenderPerfBoxplot(perfGender):
    boxplot = Boxplot(init_opts=opts.InitOpts(width='300px',height='500px'))
    boxplot=(
        boxplot.add_xaxis(['F','M'])
        .add_yaxis(
            'Performance Boxplot',
            # timeExcercisesF,
            boxplot.prepare_data(perfGender)
            )
        .set_global_opts(
            legend_opts=opts.LegendOpts(is_show=False),
            # yaxis_opts=opts.AxisOpts(is_show=False),
            # xaxis_opts=opts.AxisOpts(is_show=False),
            # tooltip_opts=opts.TooltipOpts(is_show=False)
        )
    )
    return boxplot

def getRacePerfBoxplot(perfRace):
    boxplot = Boxplot(init_opts=opts.InitOpts(width='300px',height='500px'))
    boxplot=(
        boxplot.add_xaxis(['Caucasian','African American','Asian','Multi-Race','American Indian','Unknown','Hawaiian or other'])
        .add_yaxis(
            'Performance Boxplot',
            # timeExcercisesF,
            boxplot.prepare_data(perfRace)
            )
        .set_global_opts(
            legend_opts=opts.LegendOpts(is_show=False),
            # yaxis_opts=opts.AxisOpts(is_show=False),
            # xaxis_opts=opts.AxisOpts(is_show=False),
            # tooltip_opts=opts.TooltipOpts(is_show=False)
        )
    )
    return boxplot

def getHispPerfBoxplot(perfHisp):
    boxplot = Boxplot(init_opts=opts.InitOpts(width='300px',height='500px'))
    boxplot=(
        boxplot.add_xaxis(['Y','N'])
        .add_yaxis(
            'Performance Boxplot',
            # timeExcercisesF,
            boxplot.prepare_data(perfHisp)
            )
        .set_global_opts(
            legend_opts=opts.LegendOpts(is_show=False),
            # yaxis_opts=opts.AxisOpts(is_show=False),
            # xaxis_opts=opts.AxisOpts(is_show=False),
            # tooltip_opts=opts.TooltipOpts(is_show=False)
        )
    )
    return boxplot