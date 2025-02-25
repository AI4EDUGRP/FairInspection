from pyecharts.charts import Sankey
import pyecharts.options as opts
from pyecharts.globals import ThemeType

# questions sankey diagram
def question_practice_sankey(colors,nodes,links):    
    sankey = (
        Sankey(init_opts=opts.InitOpts(
            width='1400',
            height='800px',
            theme=ThemeType.LIGHT))
        .set_colors(colors)
        .add(
            "",
            nodes=nodes,
            links=links,
            pos_bottom="10%",
            focus_node_adjacency="allEdges",
            orient="vertical",
            linestyle_opt=opts.LineStyleOpts(opacity=0.6, curve=0.5, color="source"),
            label_opts=opts.LabelOpts(position="top"),
            pos_top='10%'
        )
        .set_global_opts(
            # title_opts=opts.TitleOpts(title="Students Practice Data",
            #                          subtitle='This compares the differences in practices on the modules and questions between male and female students',
            #                          pos_left='center'),
            legend_opts=opts.LegendOpts(pos_left='None'),
            tooltip_opts=opts.TooltipOpts(trigger="item", trigger_on="mousemove"),

        )
    )
    return sankey
