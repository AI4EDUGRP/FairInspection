from pyecharts.charts import *
import pyecharts.options as opts
from pyecharts.options.global_options import ThemeType

             
def get_section_overlap_gender(section,featureDictM,featureDictF) -> Bar:
    featureList=['question_diversity', 'times_excercises', 'activeness', 'correct_nums', 'average_time']
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
    )
    return bar

# sections = ['Section 1','Section 2', 'Section 3', 'Section 4', 'Section 7', 'Section 8','Section 9', 'Section 10', 'Section 11', 'Section 12']
def getTimelineGender(sections,featureDict1,featureDict2):
    timeline = Timeline(init_opts=opts.InitOpts(
        width='500px',
        height='500px'))
    for section in sections:
        timeline.add(get_section_overlap_gender(section,featureDict1,featureDict2),time_point=section)
        timeline.add_schema(is_auto_play=True,play_interval=1000)
    return timeline



def get_section_overlap_race(section,featureDict1,featureDict2,featureDict3, featureDict4) -> Bar:
    featureList=['question_diversity', 'times_excercises', 'activeness', 'correct_nums', 'average_time']
    bar = (
        Bar(init_opts = opts.InitOpts(
             width='500px',
            height='500px',
            theme=ThemeType.ESSOS))
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

    )
    return bar

# sections = ['Section 1','Section 2', 'Section 3', 'Section 4', 'Section 7', 'Section 8','Section 9', 'Section 10', 'Section 11', 'Section 12']
def getTimelineRace(sections,featureDict1,featureDict2,featureDict3, featureDict4):
    timeline = Timeline(init_opts=opts.InitOpts(
        width='500px',
            height='500px',))
    for section in sections:
        timeline.add(get_section_overlap_race(section,featureDict1,featureDict2,featureDict3, featureDict4),time_point=section)
        timeline.add_schema(is_auto_play=True,play_interval=1000)
    return timeline