from pyecharts.charts import HeatMap, Line, Scatter, Grid
import pyecharts.options as opts
from pyecharts.globals import ThemeType


# get the section activeness
def section_practice_heatmap(xsections,ysections,data_sections):
    heatmap=(
        HeatMap(init_opts=opts.InitOpts(width='650px',height='508px'))
        .add_xaxis(xaxis_data=xsections)
        .add_yaxis(
            yaxis_data=ysections,
            series_name='Times of practice:',
            value=data_sections,
            label_opts=opts.LabelOpts(is_show=False),
            )
        .set_series_opts()
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Student Practices on each question"),
            legend_opts=opts.LegendOpts(is_show=False),
            xaxis_opts=opts.AxisOpts(
                type_='category',
                name='sections',
                name_gap=15,
                name_location='center',
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True,
                    areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                # splitline_opts=opts.SplitLineOpts(is_show=True)
            ), 
            yaxis_opts=opts.AxisOpts(
                type_='category',
                name='questions',
                name_location='center',
                name_gap=25,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True,
                    areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(is_show=True),
                axislabel_opts=opts.AxisLineOpts(is_show=True),
            ),
            visualmap_opts=opts.VisualMapOpts(
                min_=0, 
                max_=9000,
                is_calculable=True,
                orient="horizontal",
                pos_left='center',
                item_width=20,item_height=500
            ),
        )
    )
    return heatmap


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
            linestyle_opts=opts.LineStyleOpts(width=1.5),
            is_smooth=True,
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Student Performance Comparison", subtitle="The comparison of number of male and female student at different possibilities of pass the course", pos_left="center"
            ),
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
            legend_opts=opts.LegendOpts(pos_left="left",pos_top='1%'),
            toolbox_opts=opts.ToolboxOpts(
                is_show=True,
                feature={
                    "dataZoom": {"yAxisIndex": "none"},
                    "restore": {},
                    "saveAsImage": {},
                },
            ),
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
            linestyle_opts=opts.LineStyleOpts(width=1.5),
            is_smooth=True,
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
            legend_opts=opts.LegendOpts(pos_left='left',pos_top="5%"),
        )
    )

    rainfall=(
        Grid(init_opts=opts.InitOpts(width="1040px", height="600px",theme=ThemeType.DARK))
        .add(chart=l1,
             grid_opts=opts.GridOpts(pos_left=60, pos_right=60, pos_top="15%"),
            )
        .add(
            chart=l2,
            grid_opts=opts.GridOpts(pos_left=60, pos_right=60, pos_top="15%"),
        )
    )
    return rainfall


    # student performance distribution
def performance_distribution(x_data_M,y_data_M,y_data_F,size_M,size_F):
    scatter=(
        Scatter(init_opts=opts.InitOpts(width="1024px", height="768px"))
        .add_xaxis(xaxis_data=x_data_M)
        .add_yaxis('Male Students',
                   y_axis=[list(y_m) for y_m in zip(y_data_M,size_M)],
                   symbol_size=5,
                   label_opts=opts.LabelOpts(is_show=False)
                   )
        .add_yaxis('Female Students',
                   y_axis= [list(y_f) for y_f in zip(y_data_F,size_F)],
                   symbol_size=5,
                   symbol='rectangle',
                   label_opts=opts.LabelOpts(is_show=False)
           )
         .set_global_opts(
             title_opts=opts.TitleOpts(title='The distributions of student performance',
                                      subtitle="This is the comparison of the performance of male and female students and the color represents the possibilities of pass the course",
                                      pos_left='center'),
             xaxis_opts=opts.AxisOpts(
                 type_="value", 
                 ),
             yaxis_opts=opts.AxisOpts(
                 type_="value",
                 axistick_opts=opts.AxisTickOpts(is_show=True),
                 ),
             legend_opts=opts.LegendOpts(pos_left="left",pos_top='1%'),
             visualmap_opts=opts.VisualMapOpts(
                 min_=0, 
                 max_=1, 
                 is_calculable=True, 
                 orient="vertical",
                 pos_left=20,
                 is_show=True,
                 ),
            #  tooltip_opts=opts.TooltipOpts(formatter=JsCode(
            #      "function (params){return 'Probability of pass: </br>'+params.value[2]}"
            #  )),
             
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_show=True,
                    is_realtime=True,
                    start_value=30,
                    end_value=70,
                )
            ],
         )
    )
    return scatter