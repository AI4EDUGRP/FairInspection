# from re import M
# from turtle import width
# from anyio import open_cancel_scope
import attr
from pyecharts.charts import *
import pyecharts.options as opts
from pyecharts.options.global_options import ThemeType
from pyecharts.commons.utils import JsCode
import numpy as np

def demo_pie(title, attr_list, value_list):
    '''
        Pie plot of the demographic information distribution
        input: title 表明展示哪个feature的distribution
               attr_list 表明某一个feature 的列表
               value_list 与上述feature 列表相对应的值
    '''
    pie = (
        Pie(init_opts = opts.InitOpts(
            width='340px',
            height='200px',
            page_title = 'Demog',
            theme=ThemeType.LIGHT
        ))
        .add(
            title,
            [list(attrs) for attrs in zip(attr_list, value_list)],
            # center=["40%","50%"]
        )
        .set_global_opts(
            title_opts = opts.TitleOpts(''),
            legend_opts = opts.LegendOpts(
                type_ = "scroll", 
                orient="vertical",
                pos_left="80%"
                )
        )
        .set_series_opts(label_opts = opts.LabelOpts(formatter='{b}: {c}'))
    )

    return pie

# attr_list = ['M','F']
# attr_value = [411,500]

# demo_pie('Gender Distribution',attr_list, attr_value)

# attr_list = ['Caucasian','Black or African American','Asian','Two or More Races','American Indian','Unknown','Hawaiian or Other Pacific Islander']
# attr_value =[6759, 2445, 336, 304, 105, 29, 8]         
# pie = demo_pie('Race Distribution', attr_list, attr_value)      
# pie.render('Gender.html')             
             
def get_section_overlap_gender(section,featureDictM,featureDictF) -> Bar:
    featureList=['Question Diversity', 'Practice Times', 'Activeness', 'Correct Numbers', 'Average Time']
    bar = (
        Bar(init_opts = opts.InitOpts(theme=ThemeType.ESSOS))
        .add_xaxis(xaxis_data = featureList)
        .add_yaxis(
            series_name='M',
            y_axis=featureDictM[section]
        )
        .add_yaxis(
            series_name='F',
            y_axis = featureDictF[section]
        )
        .set_global_opts(
                yaxis_opts=opts.AxisOpts(is_show=False)
        )
    )
    return bar

# sections = ['Section 1','Section 2', 'Section 3', 'Section 4', 'Section 7', 'Section 8','Section 9', 'Section 10', 'Section 11', 'Section 12']
def getTimelineGender(sections,featureDict1,featureDict2):
    timeline = Timeline(init_opts=opts.InitOpts())
    for section in sections:
        timeline.add(get_section_overlap_gender(section,featureDict1,featureDict2),time_point=section)
        timeline.add_schema(is_auto_play=True,play_interval=1000)
    return timeline

def get_section_overlap_race(section,featureDict1,featureDict2,featureDict3, featureDict4) -> Bar:
    featureList=['Question Diversity', 'Practice Times', 'Activeness', 'Correct Numbers', 'Average Time']
    bar = (
        Bar(init_opts = opts.InitOpts(theme=ThemeType.ESSOS))
        .add_xaxis(xaxis_data = featureList)
        .add_yaxis(
            series_name='American Indian',
            y_axis=featureDict1[section]
        )
        .add_yaxis(
            series_name='Asian',
            y_axis = featureDict2[section]
        )
        .add_yaxis(
            series_name='Black or African American',
            y_axis = featureDict3[section]
        )
        .add_yaxis(
            series_name='Caucasian',
            y_axis = featureDict4[section]
        )        
        .set_global_opts(
                yaxis_opts=opts.AxisOpts(is_show=False)
        )
    )
    return bar

# sections = ['Section 1','Section 2', 'Section 3', 'Section 4', 'Section 7', 'Section 8','Section 9', 'Section 10', 'Section 11', 'Section 12']
def getTimelineRace(sections,featureDict1,featureDict2,featureDict3, featureDict4):
    timeline = Timeline(init_opts=opts.InitOpts())
    for section in sections:
        timeline.add(get_section_overlap_race(section,featureDict1,featureDict2,featureDict3, featureDict4),time_point=section)
        timeline.add_schema(is_auto_play=True,play_interval=1000)
    return timeline


def get_section_overlap_hisp(section,featureDictM,featureDictF) -> Bar:
    featureList=['Question Diversity', 'Practice Times', 'Activeness', 'Correct Numbers', 'Average Time']
    bar = (
        Bar(init_opts = opts.InitOpts(theme=ThemeType.ESSOS))
        .add_xaxis(xaxis_data = featureList)
        .add_yaxis(
            series_name='Y',
            y_axis=featureDictM[section]
        )
        .add_yaxis(
            series_name='N',
            y_axis = featureDictF[section]
        )
        .set_global_opts(
            yaxis_opts=opts.AxisOpts(is_show=False)
        )
    )
    return bar

# sections = ['Section 1','Section 2', 'Section 3', 'Section 4', 'Section 7', 'Section 8','Section 9', 'Section 10', 'Section 11', 'Section 12']
def getTimelineHisp(sections,featureDict1,featureDict2):
    timeline = Timeline(init_opts=opts.InitOpts())
    for section in sections:
        timeline.add(get_section_overlap_hisp(section,featureDict1,featureDict2),time_point=section)
        timeline.add_schema(is_auto_play=True,play_interval=1000)
    return timeline


#############################################################################################################
### Visualizing the correlations between features that are separated by the demographic information--Bar
#############################################################################################################

def featureDistBar(barsF,barsM,nonSensFeature,numBins=10)->Bar:
    x_axis = list(range(min(len(barsF),len(barsM))))[:numBins]
    bar = (
        Bar(init_opts=opts.InitOpts(width='500px'))
        .add_xaxis(x_axis)
        .add_yaxis(
            'F',
            barsF,
            gap = '-100%',
            itemstyle_opts=opts.ItemStyleOpts(
                # opacity=0.7,
                )
        )
        .add_yaxis(
            'M',
            barsM,
            gap = '-85%',
            itemstyle_opts=opts.ItemStyleOpts(
                # opacity=0.75,
                )
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(opacity=0.75)
        )
        .set_global_opts(
            title_opts = opts.TitleOpts(title=nonSensFeature),
            xaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(is_show=False),
                axistick_opts=opts.AxisTickOpts(is_show=False)
            ),
            yaxis_opts = opts.AxisOpts(
                name = "Number of Students",
                name_location= 'center',
                name_gap=40,
                name_textstyle_opts= opts.TextStyleOpts(
                    font_family='Arial',
                    font_size= 14
                )
                ),
            tooltip_opts = opts.TooltipOpts(formatter="{a}: {c}")
        )     
    )
    return bar

##############################################################################################################
### Visualizing the correlations between features that are separated by the demographic information--Line
##############################################################################################################
def featureDistLine(lineF,lineM,nonSensFeature,stack='stacked',numSamples=100,data=None)->Line:
    xaxis = list(range(max(len(lineF),len(lineM))))[:numSamples]
    line=(
        Line()
        .add_xaxis(
            xaxis_data=xaxis
        )
        .add_yaxis(
            series_name='F',
            y_axis=lineF,
            stack=stack,
            areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False),
            is_smooth=True
        )
        .add_yaxis(
            series_name='M',
            y_axis=lineM,
            stack=stack,
            areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
            label_opts=opts.LabelOpts(is_show=False),
            is_smooth=True
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title=' '.join(nonSensFeature.split('_')).capitalize()),
            tooltip_opts = opts.TooltipOpts(trigger='axis'),
            # tooltip_opts=opts.TooltipOpts(formatter="{a}: {c}"),
            xaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(is_show=False),
                axistick_opts=opts.AxisTickOpts(is_show=False)
            ),
            legend_opts=opts.LegendOpts(pos_top='15%',pos_left='left')
        )   
    )
    if data is not None:
        nonSensF =[[float(val) for val in data[data.gender=='F'][nonSensFeature].values]]
        nonSensM =[[float(val) for val in data[data.gender=='M'][nonSensFeature].values]]

        boxplot = Boxplot()
        boxplot=(
            boxplot.add_xaxis(['F','M'])
            .add_yaxis(
                'F',
                # timeExcercisesF,
                boxplot.prepare_data(nonSensF)
                )
            .add_yaxis(
                'M',
                # timeExcercisesM,
                boxplot.prepare_data(nonSensM),
                )
            .set_global_opts(
                legend_opts=opts.LegendOpts(is_show=False),
                yaxis_opts=opts.AxisOpts(is_show=False),
                xaxis_opts=opts.AxisOpts(is_show=False),
                tooltip_opts=opts.TooltipOpts(trigger='axis')
            )

        )
        line=(
            Grid(init_opts=opts.InitOpts(width='350px',height='235px'))
            .add(
                line,
                grid_opts=opts.GridOpts(
                    pos_bottom='0%'
                    )
                )
            .add(
                boxplot,
                grid_opts=opts.GridOpts(
                    pos_bottom='60%',
                    pos_left='60%'
                )
            )
        )
    return line


# 实现对所有的学生的学习成绩的统计根据所Histogram的分割方法
def perfDist(meanVal,numStud,maxNum=100):
    '''
    params:
        meanVal表示由Histogram所分割的各个小bins的均值
        numStud表示分配到这个bin区间内的学生的人数
    '''
    perfLine=(
        Line(init_opts = opts.InitOpts(
            width='340px',
            height='200px',
        ))
        .add_xaxis(xaxis_data=np.round(meanVal,3))
        .add_yaxis(
            series_name='Student Number',
            y_axis=numStud,
            is_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
            areastyle_opts=opts.AreaStyleOpts(opacity=.1,color='#C67570'),
            is_smooth=True
        )
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(
                type_='value',
                min_=0.01,
                max_=0.99,
                name = 'possibility of passing the course',
                name_location = 'center',
                name_gap= 25,
            
            ),
            yaxis_opts=opts.AxisOpts(
                type_='value',
                min_=0,
                # max_=maxNum,
                # name = 'Student Number',
                name_location= 'end',
                name_gap=25
            ),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts = opts.LegendOpts(is_show=False),
            datazoom_opts=[
                opts.DataZoomOpts(
                    is_realtime=True,
                    type_='inside',
                    start_value=0.45,
                    end_value=0.75
                )
            ]
        )
    )
    return perfLine

