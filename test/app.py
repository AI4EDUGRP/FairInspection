from pkgutil import get_data
import pandas as pd
import numpy as np
import json
import os

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns',None)

from pyecharts import options as opts
from pyecharts.charts import *
from pyecharts.options.global_options import ThemeType
from pyecharts.commons.utils import JsCode
from pyecharts.globals import CurrentConfig,NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
from pyecharts.globals import ThemeType
from jinja2 import Markup, Environment, FileSystemLoader

from utils import sankey_diagram_for_section_questions

CurrentConfig.GLOBAL_ENV=Environment(FileSystemLoader('./templates'))


from flask import render_template
import flask

dataPath="../../alg1_data/visualization"
dataset='../../alg1_data/dataset.csv'
dataset=pd.read_csv(dataset)

def read_data(fileName):
    path=os.path.join(dataPath,fileName)
    with open(path) as f:
        data=json.load(f)
    f.close()
    return data


scaler=StandardScaler()
data=scaler.fit_transform(dataset.iloc[:,4:-1])
X_embedded=TSNE(n_components=2,init='random').fit_transform(data)

stud_sect_ques_lists=[]
dataPath1='../../alg1_data/section_question_tb'
for sect in ['Section_1.csv','Section_2.csv','Section_3.csv','Section_4.csv','Section_7.csv','Section_8.csv','Section_9.csv','Section_10.csv','Section_11.csv','Section_12.csv']:
    stud_sect_ques_lists.append(pd.read_csv(os.path.join(dataPath1,sect)).set_index('useraccount_id'))


app = flask.Flask(__name__,static_folder='templates')

## Sankey diagram

links=[]
nodes=[]
colors=[]


## Demographic distributions
def demo_distribution(Gender,gender_num,race,race_num):
    pieGender = (
        Pie()
        .add(
            "",
            [list(gender) for gender in zip(Gender, gender_num)],
            radius=["10%", "65%"],
            center=["25%", "60%"],
            rosetype="radius",
            label_opts=opts.LabelOpts(is_show=True),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Student distributions in gender",
                                     pos_left='2%',
                                     ),
            legend_opts=opts.LegendOpts(pos_left='2%',pos_top='6%')
        )
    )

    pieRace = (
        Pie()
        .add(
            "",
            [list(h) for h in zip(race, race_num)],
            radius=["10%", "65%"],
            center=["75%", "60%"],
            rosetype="radius",
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Student distributions in race",
                                     pos_right='5%'),
            legend_opts=opts.LegendOpts(pos_right='2%',pos_top='5%',pos_left='55%')
        )
    )

    gridDemo=Grid(init_opts=opts.InitOpts(width='1040px',theme=ThemeType.DARK))
    gridDemo.add(pieGender,
         grid_opts=opts.GridOpts(pos_bottom='5%')
        )
    gridDemo.add(pieRace,
        grid_opts=opts.GridOpts(pos_bottom='5%')
        )
    
    return gridDemo

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

# get the section activeness
def section_practice_heatmap(xsections,ysections,data_sections):
    heatmap=(
        HeatMap(init_opts=opts.InitOpts(width='1440px',height='720px'))
        .add_xaxis(xaxis_data=xsections)
        .add_yaxis(
            yaxis_data=ysections,
            series_name='Times of practice:',
            value=data_sections,
            label_opts=opts.LabelOpts(is_show=False),
            )
        .set_series_opts()
        .set_global_opts(
            title_opts=opts.TitleOpts(title='The Heatmap of Practice on Each Question',
                                     subtitle='The number of practices on each question',
                                     pos_left='center'),
            legend_opts=opts.LegendOpts(is_show=False,
                                       pos_left='5%'),
            xaxis_opts=opts.AxisOpts(
                type_='category',
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True,
                    areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                # splitline_opts=opts.SplitLineOpts(is_show=True)
            ), 
            yaxis_opts=opts.AxisOpts(
                type_='category',
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True,
                    areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
                axistick_opts=opts.AxisTickOpts(is_show=False),
            ),
            visualmap_opts=opts.VisualMapOpts(
                min_=0, 
                max_=9000,
                is_calculable=True,
                orient="horizontal",
                pos_left='center',
                item_width=20,item_height=800
            ),
        )
    )
    return heatmap

# performance comparison
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


@app.route("/",methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/sankey')
def get_sankey():
    links = read_data('linksdf.json')
    nodes = read_data('nodesdf.json')
    colors = read_data('colorsdf.json')

    sankey = sankey_diagram_for_section_questions(colors,nodes,links)
    # return sankey.dump_options_with_quotes()
    return render_template('index.html',sankey=Markup(sankey))

@app.route('/stud_dist')
def get_studDist():
    x_data_M=[np.round(float(x),3) for x in X_embedded[:,0]]
    y_data_M=[np.round(float(x),3) for x in X_embedded[(dataset.gender=='M')][:,1]]
    y_data_F=[np.round(float(x),3) for x in X_embedded[(dataset.gender=='F')][:,1]]

    size_M=[np.round(float(s),3) for s in dataset[(dataset.gender=='M')].performance.values]
    size_F=[np.round(float(s),3) for s in dataset[(dataset.gender=='F')].performance.values]
    stud_dist=performance_distribution(x_data_M,y_data_M,y_data_F,size_M,size_F)

    return stud_dist.dump_options_with_quotes()

@app.route('/demo_dist')
def get_demog():
    Gender=['Male','Female']
    gender_num=[4715, 5271]

    race=['Caucasian','Black or African American','Asian','Two or More Races','American Indian','Unknown','Hawaiian or Other Pacific Islander']
    race_num=[6759,2445,336,304,105,29,8]
    demog_grid=demo_distribution(Gender,gender_num,race,race_num)

    return demog_grid.dump_options_with_quotes()

@app.route('/sect_dist')
def get_sect():
    xsections=['' for i in range(68)]
    ysections=['' for i in range(8)]
    r1=0
    r2=0
    data_sections=[]
    data_questions=[]
    for ind in [0,1,2,4,3,6,7,9,8,5]:
        sect_ques_visit=stud_sect_ques_lists[ind].iloc[:,:-3].sum().values
        sect_ques_ids=list(stud_sect_ques_lists[ind].iloc[:,:-3].sum().index)
        s=len(sect_ques_visit)
        r2+=int(np.ceil(s/8))
        data_sect=[]
        ids=0
        for i in range(r1,r2):
            for j in range(8):
                if ids>s-1:
                    val=0
                else:
                    val=sect_ques_visit[ids]
                data_sect.append([i,j,val])
                ids+=1
                data_questions.extend(sect_ques_ids)
        r1=r2
        data_sections.extend(data_sect)
    sect_dist=section_practice_heatmap(xsections,ysections,data_sections)
    
    return sect_dist.dump_options_with_quotes()

@app.route('/perf_dist')
def get_perf():
    M_perfs=dataset[dataset.gender=='M'].performance.values
    F_perfs=dataset[dataset.gender=='F'].performance.values

    M_values,M_bins=np.histogram(M_perfs,bins=1000)
    M_values=list(M_values)
    M_bins=list(M_bins)
    M_values.insert(0,0)

    F_values,F_bins=np.histogram(F_perfs,bins=1000)
    F_values=list(F_values)
    F_bins=list(F_bins)
    F_values.insert(0,0)

    bins= M_bins
    female_performance=F_values
    male_performance=M_values

    bins=[str(np.round(t,3)) for t in bins]
    female_performance=[float(f) for f in female_performance]
    male_performance=[float(f) for f in male_performance]

    perf_dist = performance_rainfall(bins,male_performance,female_performance)
    return perf_dist.dump_options_with_quotes()


if __name__=="__main__":
    app.run(debug=True)


