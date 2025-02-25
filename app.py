import pickle
import pandas as pd
import numpy as np
import os
import sklearn 
from sklearn import neighbors

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from utils.fairnessVisualization import getThresholdFair
pd.set_option('display.max_columns',None)

from pyecharts import options as opts
from pyecharts.charts import *
from pyecharts.options.global_options import ThemeType
from pyecharts.commons.utils import JsCode
from pyecharts.globals import CurrentConfig,NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
from pyecharts.globals import ThemeType
from jinja2 import Environment, FileSystemLoader

from utils.sankey_diagram_for_section_questions import *
from utils.visualization import *
from utils.featureCorrelation import *
from utils.dataProcessing import *
from utils.student_practice import *
from utils.rainFall import *
from utils.sankey_diagram_for_section_questions import *
from utils.mlVisual import *
from utils.modelMetricVisualization import *
from utils.fairnessVisualization import *

from ml.modelExplainer import *
from ml.nearestNeighbor import *

from fairness.fairnessEvaluation import *
from fairness.biasMitigation import *

CurrentConfig.GLOBAL_ENV=Environment(loader=FileSystemLoader('./templates')) #注意要把loader加上

from flask import render_template, url_for,request
import flask


# 读入原数据集
dataset=pd.read_csv('data/dataset.csv')
# demographic information distribution
pie_gender = demo_pie('Gender Distribution', list(dataset.gender.value_counts().index), [int(value) for value in dataset.gender.value_counts().values])
pie_hispanic = demo_pie('hispanic Distribution', list(dataset.hispanic_ethnicity.value_counts().index), [int(value) for value in dataset.hispanic_ethnicity.value_counts().values])
pie_race = demo_pie('race Distribution', list(dataset.race.value_counts().index), [int(value) for value in dataset.race.value_counts().values])

# question practice heatmap
data_sections, xsections, ysections = prepareHeatmap()
heatmap = section_practice_heatmap(xsections,ysections,data_sections)

# get the student performance distribution
val, nums = getPerDist(dataset)
perfLine = perfDist(val,nums,maxNum=400)

#get student rainFall performance
bins, male_performance, female_performance = getGenderPerf(dataset)
rainFallDist = performance_rainfall(bins,male_performance,female_performance)

bins, hisp_performance, nonHisp_performance = getHispPerf(dataset)
rainFallDistHisp = performance_rainfall(bins,hisp_performance,nonHisp_performance)

#get student performance boxplot
perfGender = getGenderPerfBoxplotData(dataset)
genderPerfBoxplot=getGenderPerfBoxplot(perfGender)

perfRace = getRacePerfBoxplotData(dataset)
racePerfBoxplot=getRacePerfBoxplot(perfRace)

perfHisp = getHispPerfBoxplotData(dataset)
hispPerfBoxplot=getHispPerfBoxplot(perfHisp)

featureDictF, featureDictM, sections = getTimelineData(dataset,'gender')
timelineGender = getTimelineGender(sections,featureDictF,featureDictM)

featureDictAmericanIndian,featureDictAsian,featureDictBAA,featureDictCaucasian,sections = getTimelineData(dataset,'race')
timelineRace = getTimelineRace(sections,featureDictAmericanIndian,featureDictAsian,featureDictBAA, featureDictCaucasian)

featureDictHisp, featureDictNonHisp, sections = getTimelineData(dataset,'hispanic_ethnicity')
timelineHisp = getTimelineHisp(sections,featureDictHisp,featureDictNonHisp)

# get sankey graph
links, nodes, colors = getSankeyData('data/visualization')
questionSankey=question_practice_sankey(colors,nodes,links)

###############################################################################################################################
###### Student Activity Data Averaged on Each Module
###############################################################################################################################
studModuleActDf = pd.read_csv('data/studAggregatedDataUpdated.csv')
# barF,barM=nonSensFeatDistGender(studModuleActDf,nonSensFeature='time_excercises')
# barTime = featureDistBar(barF,barM,'time_excercises')

lineF,lineM=nonSensFeatDistGender(studModuleActDf,nonSensFeature='question_diversity',bin_size_calculator=customizeBinSize)
lineDiversity = featureDistLine(lineF,lineM,'question_diversity',stack=None,data=studModuleActDf)

lineF,lineM=nonSensFeatDistGender(studModuleActDf,nonSensFeature='time_excercises',bin_size_calculator=customizeBinSize)
lineActiveness = featureDistLine(lineF,lineM,'time_excercises',stack=None,data=studModuleActDf)

lineF,lineM=nonSensFeatDistGender(studModuleActDf,nonSensFeature='active_days',bin_size_calculator=customizeBinSize)
lineActDays = featureDistLine(lineF,lineM,'active_days',stack=None,data=studModuleActDf)

lineF,lineM=nonSensFeatDistGender(studModuleActDf,nonSensFeature='correct_nums',bin_size_calculator=customizeBinSize)
lineCorrNums = featureDistLine(lineF,lineM,'correct_nums',stack=None,data=studModuleActDf)

lineF,lineM=nonSensFeatDistGender(studModuleActDf,nonSensFeature='average_time',bin_size_calculator=customizeBinSize)
lineAvgTime = featureDistLine(lineF,lineM,'average_time',stack=None,data=studModuleActDf)

#################################################################################################
## Machine Learning models
################################################################################################
studDfTruePass_lr = pd.read_csv('data/lrModel/studDfTruePass.csv')
studDfTrueFail_lr = pd.read_csv('data/lrModel/studDfTrueFail.csv')
studDfPredPass_lr = pd.read_csv('data/lrModel/studDfPredPass.csv')
studDfPredFail_lr = pd.read_csv('data/lrModel/studDfPredFail.csv')
studDfFNS_lr = pd.read_csv('data/lrModel/studDfFNS.csv')
studDfFPS_lr = pd.read_csv('data/lrModel/studDfFPS.csv')
perfTrue_lr = StudentTrueScatter(studDfTruePass_lr, studDfTrueFail_lr)
perfPred_lr = StudentPredScatter(studDfPredPass_lr, studDfPredFail_lr, studDfFNS_lr, studDfFPS_lr)


studDfTruePass_svc = pd.read_csv('data/svcModel/studDfTruePass.csv')
studDfTrueFail_svc = pd.read_csv('data/svcModel/studDfTrueFail.csv')
studDfPredPass_svc = pd.read_csv('data/svcModel/studDfPredPass.csv')
studDfPredFail_svc = pd.read_csv('data/svcModel/studDfPredFail.csv')
studDfFNS_svc = pd.read_csv('data/svcModel/studDfFNS.csv')
studDfFPS_svc = pd.read_csv('data/svcModel/studDfFPS.csv')
perfTrue_svc = StudentTrueScatter(studDfTruePass_svc, studDfTrueFail_svc)
perfPred_svc = StudentPredScatter(studDfPredPass_svc, studDfPredFail_svc, studDfFNS_svc, studDfFPS_svc)


studDfTruePass_dt = pd.read_csv('data/dtModel/studDfTruePass.csv')
studDfTrueFail_dt = pd.read_csv('data/dtModel/studDfTrueFail.csv')
studDfPredPass_dt = pd.read_csv('data/dtModel/studDfPredPass.csv')
studDfPredFail_dt = pd.read_csv('data/dtModel/studDfPredFail.csv')
studDfFNS_dt = pd.read_csv('data/dtModel/studDfFNS.csv')
studDfFPS_dt = pd.read_csv('data/dtModel/studDfFPS.csv')
perfTrue_dt = StudentTrueScatter(studDfTruePass_dt, studDfTrueFail_dt)
perfPred_dt = StudentPredScatter(studDfPredPass_dt, studDfPredFail_dt, studDfFNS_dt, studDfFPS_dt)

user_test = pd.read_csv('data/preprocessData/user_test.csv')
default_user = user_test.sample(1)['useraccount_id'].values[0]

debiased_df = pd.read_csv('fairness/debiased_df.csv')
exp_grad_red_lr = pd.read_csv('fairness/exp_grad_red_lr.csv')
exp_grad_red_dt = pd.read_csv('fairness/exp_grad_red_dt.csv')

roc_lr_df = pd.read_csv('fairness/roc_lr_df.csv')
roc_svc_df = pd.read_csv('fairness/roc_svc_df.csv')
roc_dt_df = pd.read_csv('fairness/roc_dt_df.csv')

cpp_lr_df = pd.read_csv('fairness/cpp_lr_df.csv')
cpp_svc_df = pd.read_csv('fairness/cpp_svc_df.csv')
cpp_dt_df = pd.read_csv('fairness/cpp_dt_df.csv')


def genderMap(ind):
    if ind == 0:
        return 'M'
    else:
        return 'F'

app = flask.Flask(__name__,static_url_path='')
app.jinja_env.variable_start_string = '{{'
app.jinja_env.variable_end_string = '}}'

@app.route('/')
@app.route('/index.html')
def index():
    return render_template(
        'index.html',
        # sankey=Markup(sankey.render_embed()),
        genderDist = Markup(pie_gender.render_embed()),
        hispanicDist = Markup(pie_hispanic.render_embed()),
        raceDist = Markup(pie_race.render_embed()),
        perfDist = Markup(perfLine.render_embed()),
        rainFallDist = Markup(rainFallDist.render_embed()),
        rainFallDistHisp = Markup(rainFallDistHisp.render_embed()),
        boxplotGender=Markup(genderPerfBoxplot.render_embed()),
        boxplotRace = Markup(racePerfBoxplot.render_embed()),
        boxplotHisp = Markup(hispPerfBoxplot.render_embed()),
        sankeyQuestion = Markup(questionSankey.render_embed()),
        # genderTimeline=Markup(timelineGender.render_embed()),
        genderTimeline = Markup(timelineGender.render_embed()),
        timelineRace = Markup(timelineRace.render_embed()),
        timelineHisp = Markup(timelineHisp.render_embed()),
        quesDist = Markup(heatmap.render_embed()),
        # vals=vals
        )

@app.route('/model_lr.html',methods=['GET','POST'])
def chart():
    userIDs=[]
    userGenders=[]
    userTruePerfs=[]
    userTrueLabels=[]
    userPredPerfs=[]
    userPredLabels=[]
    userFlags=[]
    userId=0

    data = 'data/lrModel/studDf.csv'

    if request.method=='POST':
        userId=int(request.form.get('username'))
        neighborsTable, _ , _, _= getNeigborsTable(userId,6,data)
        neighborsTable.fillna('Correctly Predicted',inplace=True)
    else:
        userId=default_user
        neighborsTable, _, _, _ = getNeigborsTable(userId,6,data)
        neighborsTable.fillna('Correctly Predicted',inplace=True)

    for i in range(neighborsTable.shape[0]):
        userIDs.append(neighborsTable.loc[i,'useraccount_id'])
        userGenders.append(list(map(genderMap,[neighborsTable.loc[i,'gender']]))[0])
        userTruePerfs.append(neighborsTable.loc[i,'y_true'])
        userTrueLabels.append(neighborsTable.loc[i,'y_label'])
        userPredPerfs.append(neighborsTable.loc[i,'y_pred'])
        userPredLabels.append(neighborsTable.loc[i,'pred_label'])
        userFlags.append(neighborsTable.loc[i,'Flag'])
  
    model = pickle.load(open('data/lrModel/lr_predictor.pkl','rb'))['classification']
    # model=LogisticRegression(solver='liblinear',fit_intercept=True)
    exp_html=getExpHtml(model, userId, data)

    # shap_values,neighbors=getModelExplainerShap(model, userId)
    shap_values,explainerShap,neighbors,cols=getModelExplainerShap(model, userId, data)
    shap_plots = {}

    for i in range(len(shap_values)):
        ind = i
        shap_plots[i] = force_plot_html(shap_values, explainerShap, ind, cols, str(neighbors[i]))
        # currUser.append(neighbor)

    fpr, tpr, cmatrix, acc, prec, recall, f1score = getMetrics(data)
    barMetrics = getBarMetrics(acc,prec,recall,f1score)
    confusionmatrix = confusionMatrix(cmatrix)
    rocplot=plotROC(fpr,tpr)

    return render_template(
        'model_lr.html',
        lineDiversity = Markup(lineDiversity.render_embed()),
        lineActiveness = Markup(lineActiveness.render_embed()),
        lineActDays = Markup(lineActDays.render_embed()),
        lineCorrNums = Markup(lineCorrNums.render_embed()),
        lineAvgTime = Markup(lineAvgTime.render_embed()),
    
        perfTrue = Markup(perfTrue_lr.render_embed()),
        perfPred = Markup(perfPred_lr.render_embed()),
        userId1 = userIDs[0],
        userGender1=userGenders[0],
        userTruePerf1=userTruePerfs[0],
        userTrueLabel1=userTrueLabels[0],
        userPredPerf1=userPredPerfs[0],
        userPredLabel1=userPredLabels[0],
        userFlag1 = userFlags[0],

        userId = userId,
        userId2 = userIDs[1],
        userGender2=userGenders[1],
        userTruePerf2=userTruePerfs[1],
        userTrueLabel2=userTrueLabels[1],
        userPredPerf2=userPredPerfs[1],
        userPredLabel2=userPredLabels[1],
        userFlag2 = userFlags[1],

        userId3 = userIDs[2],
        userGender3=userGenders[2],
        userTruePerf3=userTruePerfs[2],
        userTrueLabel3=userTrueLabels[2],
        userPredPerf3=userPredPerfs[2],
        userPredLabel3=userPredLabels[2],
        userFlag3 = userFlags[2],

        userId4 = userIDs[3],
        userGender4=userGenders[3],
        userTruePerf4=userTruePerfs[3],
        userTrueLabel4=userTrueLabels[3],
        userPredPerf4=userPredPerfs[3],
        userPredLabel4=userPredLabels[3],
        userFlag4 = userFlags[3],

        userId5 = userIDs[4],
        userGender5=userGenders[4],
        userTruePerf5=userTruePerfs[4],
        userTrueLabel5=userTrueLabels[4],
        userPredPerf5=userPredPerfs[4],
        userPredLabel5=userPredLabels[4],
        userFlag5 = userFlags[4],

        userId6 = userIDs[5],
        userGender6=userGenders[5],
        userTruePerf6=userTruePerfs[5],
        userTrueLabel6=userTrueLabels[5],
        userPredPerf6=userPredPerfs[5],
        userPredLabel6=userPredLabels[5],
        userFlag6 = userFlags[5],

        barMetrics = Markup(barMetrics.render_embed()),
        confusionMatrix = Markup(confusionmatrix.render_embed()),
        rocplot = Markup(rocplot.render_embed()),
        exp = exp_html,
        shap_plots = shap_plots,        
    )

@app.route('/model_svm.html',methods=['GET','POST'])
def chart_svc():
    userIDs=[]
    userGenders=[]
    userTruePerfs=[]
    userTrueLabels=[]
    userPredPerfs=[]
    userPredLabels=[]
    userFlags=[]
    userId=0

    data = 'data/svcModel/studDf.csv'

    if request.method=='POST':
        userId=int(request.form.get('username'))
        neighborsTable, _ , _, _= getNeigborsTable(userId,6,data)
        neighborsTable.fillna('Correctly Predicted',inplace=True)
    else:
        userId=default_user
        neighborsTable, _, _, _ = getNeigborsTable(userId,6,data)
        neighborsTable.fillna('Correctly Predicted',inplace=True)

    for i in range(neighborsTable.shape[0]):
        userIDs.append(neighborsTable.loc[i,'useraccount_id'])
        userGenders.append(list(map(genderMap,[neighborsTable.loc[i,'gender']]))[0])
        userTruePerfs.append(neighborsTable.loc[i,'y_true'])
        userTrueLabels.append(neighborsTable.loc[i,'y_label'])
        userPredPerfs.append(neighborsTable.loc[i,'y_pred'])
        userPredLabels.append(neighborsTable.loc[i,'pred_label'])
        userFlags.append(neighborsTable.loc[i,'Flag'])
  
    model = pickle.load(open('data/svcModel/svc_predictor.pkl','rb'))['classification']
    # model=LogisticRegression(solver='liblinear',fit_intercept=True)
    exp_html=getExpHtml(model, userId, data)

    # shap_values,neighbors=getModelExplainerShap(model, userId)
    shap_values,explainerShap,neighbors,cols=getModelExplainerShapSVCDT(model, userId, data)
    shap_plots = {}

    for i in range(len(shap_values)):
        ind = i
        shap_plots[i] = svc_dt_force_plot_html(shap_values, explainerShap, ind, cols, str(neighbors[i]))
        # currUser.append(neighbor)

    fpr, tpr, cmatrix, acc, prec, recall, f1score = getMetrics(data)
    barMetrics = getBarMetrics(acc,prec,recall,f1score)
    confusionmatrix = confusionMatrix(cmatrix)
    rocplot=plotROC(fpr,tpr)

    return render_template(
        'model_svm.html',
        lineDiversity = Markup(lineDiversity.render_embed()),
        lineActiveness = Markup(lineActiveness.render_embed()),
        lineActDays = Markup(lineActDays.render_embed()),
        lineCorrNums = Markup(lineCorrNums.render_embed()),
        lineAvgTime = Markup(lineAvgTime.render_embed()),
    
        perfTrue = Markup(perfTrue_svc.render_embed()),
        perfPred = Markup(perfPred_svc.render_embed()),
        userId1 = userIDs[0],
        userGender1=userGenders[0],
        userTruePerf1=userTruePerfs[0],
        userTrueLabel1=userTrueLabels[0],
        userPredPerf1=userPredPerfs[0],
        userPredLabel1=userPredLabels[0],
        userFlag1 = userFlags[0],

        userId = userId,
        userId2 = userIDs[1],
        userGender2=userGenders[1],
        userTruePerf2=userTruePerfs[1],
        userTrueLabel2=userTrueLabels[1],
        userPredPerf2=userPredPerfs[1],
        userPredLabel2=userPredLabels[1],
        userFlag2 = userFlags[1],

        userId3 = userIDs[2],
        userGender3=userGenders[2],
        userTruePerf3=userTruePerfs[2],
        userTrueLabel3=userTrueLabels[2],
        userPredPerf3=userPredPerfs[2],
        userPredLabel3=userPredLabels[2],
        userFlag3 = userFlags[2],

        userId4 = userIDs[3],
        userGender4=userGenders[3],
        userTruePerf4=userTruePerfs[3],
        userTrueLabel4=userTrueLabels[3],
        userPredPerf4=userPredPerfs[3],
        userPredLabel4=userPredLabels[3],
        userFlag4 = userFlags[3],

        userId5 = userIDs[4],
        userGender5=userGenders[4],
        userTruePerf5=userTruePerfs[4],
        userTrueLabel5=userTrueLabels[4],
        userPredPerf5=userPredPerfs[4],
        userPredLabel5=userPredLabels[4],
        userFlag5 = userFlags[4],

        userId6 = userIDs[5],
        userGender6=userGenders[5],
        userTruePerf6=userTruePerfs[5],
        userTrueLabel6=userTrueLabels[5],
        userPredPerf6=userPredPerfs[5],
        userPredLabel6=userPredLabels[5],
        userFlag6 = userFlags[5],

        barMetrics = Markup(barMetrics.render_embed()),
        confusionMatrix = Markup(confusionmatrix.render_embed()),
        rocplot = Markup(rocplot.render_embed()),
        exp = exp_html,
        shap_plots = shap_plots,        
    )

@app.route('/model_dt.html',methods=['GET','POST'])
def chart_dt():
    userIDs=[]
    userGenders=[]
    userTruePerfs=[]
    userTrueLabels=[]
    userPredPerfs=[]
    userPredLabels=[]
    userFlags=[]
    userId=0

    data = 'data/dtModel/studDf.csv'

    if request.method=='POST':
        userId=int(request.form.get('username'))
        neighborsTable, _ , _, _= getNeigborsTable(userId,6,data)
        neighborsTable.fillna('Correctly Predicted',inplace=True)
    else:
        userId=default_user
        neighborsTable, _, _, _ = getNeigborsTable(userId,6,data)
        neighborsTable.fillna('Correctly Predicted',inplace=True)

    for i in range(neighborsTable.shape[0]):
        userIDs.append(neighborsTable.loc[i,'useraccount_id'])
        userGenders.append(neighborsTable.loc[i,'gender'])
        userTruePerfs.append(neighborsTable.loc[i,'y_true'])
        userTrueLabels.append(neighborsTable.loc[i,'y_label'])
        userPredPerfs.append(neighborsTable.loc[i,'y_pred'])
        userPredLabels.append(neighborsTable.loc[i,'pred_label'])
        userFlags.append(neighborsTable.loc[i,'Flag'])
  
    model = pickle.load(open('data/dtModel/dt_predictor.pkl','rb'))['classification']
    # model=LogisticRegression(solver='liblinear',fit_intercept=True)
    exp_html=getExpHtml(model, userId, data)

    # shap_values,neighbors=getModelExplainerShap(model, userId)
    shap_values,explainerShap,neighbors,cols=getModelExplainerShapSVCDT(model, userId, data)
    shap_plots = {}

    for i in range(len(shap_values)):
        ind = i
        shap_plots[i] = svc_dt_force_plot_html(shap_values, explainerShap, ind, cols, str(neighbors[i]))
        # currUser.append(neighbor)

    fpr, tpr, cmatrix, acc, prec, recall, f1score = getMetrics(data)
    barMetrics = getBarMetrics(acc,prec,recall,f1score)
    confusionmatrix = confusionMatrix(cmatrix)
    rocplot=plotROC(fpr,tpr)

    return render_template(
        'model_dt.html',
        lineDiversity = Markup(lineDiversity.render_embed()),
        lineActiveness = Markup(lineActiveness.render_embed()),
        lineActDays = Markup(lineActDays.render_embed()),
        lineCorrNums = Markup(lineCorrNums.render_embed()),
        lineAvgTime = Markup(lineAvgTime.render_embed()),
    
        perfTrue = Markup(perfTrue_dt.render_embed()),
        perfPred = Markup(perfPred_dt.render_embed()),
        userId1 = userIDs[0],
        userGender1=userGenders[0],
        userTruePerf1=userTruePerfs[0],
        userTrueLabel1=userTrueLabels[0],
        userPredPerf1=userPredPerfs[0],
        userPredLabel1=userPredLabels[0],
        userFlag1 = userFlags[0],

        userId = userId,
        userId2 = userIDs[1],
        userGender2=userGenders[1],
        userTruePerf2=userTruePerfs[1],
        userTrueLabel2=userTrueLabels[1],
        userPredPerf2=userPredPerfs[1],
        userPredLabel2=userPredLabels[1],
        userFlag2 = userFlags[1],

        userId3 = userIDs[2],
        userGender3=userGenders[2],
        userTruePerf3=userTruePerfs[2],
        userTrueLabel3=userTrueLabels[2],
        userPredPerf3=userPredPerfs[2],
        userPredLabel3=userPredLabels[2],
        userFlag3 = userFlags[2],

        userId4 = userIDs[3],
        userGender4=userGenders[3],
        userTruePerf4=userTruePerfs[3],
        userTrueLabel4=userTrueLabels[3],
        userPredPerf4=userPredPerfs[3],
        userPredLabel4=userPredLabels[3],
        userFlag4 = userFlags[3],

        userId5 = userIDs[4],
        userGender5=userGenders[4],
        userTruePerf5=userTruePerfs[4],
        userTrueLabel5=userTrueLabels[4],
        userPredPerf5=userPredPerfs[4],
        userPredLabel5=userPredLabels[4],
        userFlag5 = userFlags[4],

        userId6 = userIDs[5],
        userGender6=userGenders[5],
        userTruePerf6=userTruePerfs[5],
        userTrueLabel6=userTrueLabels[5],
        userPredPerf6=userPredPerfs[5],
        userPredLabel6=userPredLabels[5],
        userFlag6 = userFlags[5],

        barMetrics = Markup(barMetrics.render_embed()),
        confusionMatrix = Markup(confusionmatrix.render_embed()),
        rocplot = Markup(rocplot.render_embed()),
        exp = exp_html,
        shap_plots = shap_plots,        

    )     

@app.route('/fairness_lr.html',methods=['GET','POST'])
def fairnessInvestigate_lr():
    lr_model, preprorcessor, X_train, y_train, X_test, y_test = loadDataAndModels('data/lrModel/lr_predictor.pkl')
    class_metrics_score=0
    metric_score1=0
    metric_score2=0
    sen_attr=''
    metric_attr=''
    sen_attr1=''
    sen_attr2=''
    metric_fair=''

    if request.method == 'POST':
        sen_attr = request.form.get('sensitiveAttr')
        if sen_attr=='gender':
            sen_attr1 = 'Male'
            sen_attr2 = 'Female'
        elif sen_attr =='race':
            sen_attr1 ='Caucasian'
            sen_attr2= 'Others'
        else:
            sen_attr1 = 'Hispanic'
            sen_attr2 = 'Others'

        metric_attr = request.form.get('perfMetrics')
        metric_fair = request.form.get('fairMetric')
    else:
        sen_attr ='gender'
        sen_attr1 = 'Male'
        sen_attr2 = 'Female'
        metric_attr = 'Accuracy'
        metric_fair = 'Consistency'
        

    dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups = constructAIF36ODataset(X_train, y_train, X_test, y_test, sen_attr)
    
    y_train_pred, y_test_pred = getPredictions(X_train,X_test,lr_model)
    metric_train, metric_test = getClassMetricDataset(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups, y_train_pred, y_test_pred)

    fairRslt =getFairRslt(metric_test)
    
    num_instance = metric_test.num_instances()
    num_priv_instance = metric_test.num_instances(privileged = True)
    num_unpriv_instance = metric_test.num_instances(privileged = False)

    num_positive = metric_test.num_positives()
    num_priv_positive = metric_test.num_positives(privileged = True)
    num_unpriv_positive = metric_test.num_positives(privileged = False)

    sel_rate = np.round(metric_test.selection_rate(),3)
    sel_priv_rate = np.round(metric_test.selection_rate(privileged = True), 3)
    sel_unpriv_rate = np.round(metric_test.selection_rate(privileged = False), 3)

    TPR = np.round(metric_test.true_positive_rate(),3)
    TPRPriv = np.round(metric_test.true_positive_rate(privileged = True),3)
    TPRUnpriv = np.round(metric_test.true_positive_rate(privileged = False), 3)

    TNR = np.round(metric_test.true_negative_rate(),3)
    TNRPriv = np.round(metric_test.true_negative_rate(privileged = True),3)
    TNRUnpriv = np.round(metric_test.true_negative_rate(privileged = False),3)

    FPR = np.round(metric_test.false_positive_rate(),3)
    FPRPriv = np.round(metric_test.false_positive_rate(privileged = True),3)
    FPRUnpriv = np.round(metric_test.false_positive_rate(privileged = False),3)

    FNR = np.round(metric_test.false_negative_rate(),3)
    FNRPriv = np.round(metric_test.false_negative_rate(privileged = True),3)
    FNRUnpriv = np.round(metric_test.false_negative_rate(privileged = False),3)

    fairnesslvl=getFairnessMetric(metric_test,metric_fair)

    if metric_attr=='Accuracy':
        class_metrics_score = np.round(metric_test.accuracy(),3)
        metric_score1 = np.round(metric_test.accuracy(privileged=True),3)
        metric_score2 = np.round(metric_test.accuracy(privileged=False),3)
    elif metric_attr =='Recall':
        class_metrics_score = np.round(metric_test.recall(),3)
        metric_score1 = np.round(metric_test.recall(privileged=True),3)
        metric_score2 = np.round(metric_test.recall(privileged=False),3)
    elif metric_attr =='Precision':
        class_metrics_score = np.round(metric_test.precision(),3)
        metric_score1 = np.round(metric_test.precision(privileged=True),3)
        metric_score2 = np.round(metric_test.precision(privileged=False),3)
    elif metric_attr =='F1-Score':
        class_metrics_score = np.round(f1_score(metric_test),3)
        metric_score1 = np.round(f1_score(metric_test,privileged=True),3)
        metric_score2 = np.round(f1_score(metric_test,privileged=False),3)
    else:
        class_metrics_score = np.round(balanced_accuracy(metric_test),3)
        metric_score1 = np.round(balanced_accuracy(metric_test, privileged=True),3)
        metric_score2 = np.round(balanced_accuracy(metric_test, privileged=False),3)

    gaugeSPD=getFairGauge(fairRslt[0],-1,1)
    gaugeEOD=getFairGauge(fairRslt[1],-1,1)
    gaugeAOD=getFairGauge(fairRslt[2],-1,1)
    gaugeDI=getFairGauge(fairRslt[3],0,2,flow=0.4,fhigh=0.63)

    thresholdFairDf = pd.read_csv(os.path.join('fairness','thresholdFairDf_'+sen_attr+'_lr.csv'))
    bal_acc_Scatter=getThresholdFair(thresholdFairDf, 'spd','Threshold','Statistical Parity Difference')
    eq_opp_diff_Scatter=getThresholdFair(thresholdFairDf,'eq_opp_diff','Threshold','eq_opp_diff')
    error_rate_diff_Scatter = getThresholdFair(thresholdFairDf, 'avg_odds_diff','Threshold','Average odds Difference')
    consistency_Scatter = getThresholdFair(thresholdFairDf, 'dispImp','Threshold','Disparate Impact')

    _,metric_test_rw=reweighing(unprivileged_groups,privileged_groups,dataset_orig_train,dataset_orig_test,'Logistic Regression')
    fairRslt_rw =getFairRslt(metric_test_rw)

    gaugeSPD_rw=getFairGauge(fairRslt_rw[0],-1,1)
    gaugeEOD_rw=getFairGauge(fairRslt_rw[1],-1,1)
    gaugeAOD_rw=getFairGauge(fairRslt_rw[2],-1,1)
    gaugeDI_rw=getFairGauge(fairRslt_rw[3],0,2,flow=0.4,fhigh=0.63)

    prThresholdFairDf = pd.read_csv(os.path.join('fairness','prDf_'+sen_attr+'.csv'))
    spd_Scatter = getThresholdFair(prThresholdFairDf, 'Statistical Parity Difference','threshold','Statistical Parity Difference')
    pr_eq_opp_diff_Scatter = getThresholdFair(prThresholdFairDf,'Equal Opportunity Difference','threshold','Equal Opportunity Difference')
    pr_avg_odds_diff_Scatter = getThresholdFair(prThresholdFairDf, 'Average Odds Difference','threshold','Average Odds Difference')
    pr_di_Scatter = getThresholdFair(prThresholdFairDf, 'Disparat Impact','threshold','Disparat Impact')

    metric_test_eqOdds_lr= eqOddsPostProcessing(unprivileged_groups,privileged_groups,dataset_orig_test,dataset_orig_test,'Logistic Regression')
    fairRslt_eqOdds_lr = getFairRslt(metric_test_eqOdds_lr)

    gaugeSPD_eqOdds_lr=getFairGauge(fairRslt_eqOdds_lr[0],-1,1)
    gaugeEOD_eqOdds_lr=getFairGauge(fairRslt_eqOdds_lr[1],-1,1)
    gaugeAOD_eqOdds_lr=getFairGauge(fairRslt_eqOdds_lr[2],-1,1)
    gaugeDI_eqOdds_lr=getFairGauge(fairRslt_eqOdds_lr[3],0,2,flow=0.4,fhigh=0.63)

    # LFR
    lr_lfr = pickle.load(open('fairness/'+sen_attr+'_lr_lfr.pkl','rb'))
    y_test_transf_pred = lr_lfr.predict(dataset_orig_test.features)
    y_train_transf_pred = lr_lfr.predict(dataset_orig_train.features)
    _,metric_test_lfr=getClassMetricDataset(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups, y_train_transf_pred, y_test_transf_pred)

    fairRslt_lfr =getFairRslt(metric_test_lfr)

    gaugeSPD_lfr=getFairGauge(fairRslt_lfr[0],-1,1)
    gaugeEOD_lfr=getFairGauge(fairRslt_lfr[1],-1,1)
    gaugeAOD_lfr=getFairGauge(fairRslt_lfr[2],-1,1)
    gaugeDI_lfr=getFairGauge(fairRslt_lfr[3],0,2,flow=0.4,fhigh=0.63)

    # Disparate Remover
    _,metric_test_di = disparateImpactRemover(unprivileged_groups,privileged_groups,dataset_orig_train,dataset_orig_test,'Logistic Regression',repair_level=1.0)
    fairRslt_di =getFairRslt(metric_test_di)

    gaugeSPD_di=getFairGauge(fairRslt_di[0],-1,1)
    gaugeEOD_di=getFairGauge(fairRslt_di[1],-1,1)
    gaugeAOD_di=getFairGauge(fairRslt_di[2],-1,1)
    gaugeDI_di=getFairGauge(fairRslt_di[3],0,2,flow=0.4,fhigh=0.63)

    # Debiasing
    fairRslt_debias = debiased_df[sen_attr].values.tolist()
    gaugeSPD_debias=getFairGauge(fairRslt_debias[0],-1,1)
    gaugeEOD_debias=getFairGauge(fairRslt_debias[1],-1,1)
    gaugeAOD_debias=getFairGauge(fairRslt_debias[2],-1,1)
    gaugeDI_debias=getFairGauge(fairRslt_debias[3],0,2,flow=0.4,fhigh=0.63)

    # exp_grad_red
    fairRslt_egr_lr = exp_grad_red_lr[sen_attr].values.tolist()
    gaugeSPD_egr_lr=getFairGauge(fairRslt_egr_lr[0],-1,1)
    gaugeEOD_egr_lr=getFairGauge(fairRslt_egr_lr[1],-1,1)
    gaugeAOD_egr_lr=getFairGauge(fairRslt_egr_lr[2],-1,1)
    gaugeDI_egr_lr=getFairGauge(fairRslt_egr_lr[3],0,2,flow=0.4,fhigh=0.63)

    # roc
    fairRslt_roc_lr = roc_lr_df[sen_attr].values.tolist()
    gaugeSPD_roc_lr=getFairGauge(fairRslt_roc_lr[0],-1,1)
    gaugeEOD_roc_lr=getFairGauge(fairRslt_roc_lr[1],-1,1)
    gaugeAOD_roc_lr=getFairGauge(fairRslt_roc_lr[2],-1,1)
    gaugeDI_roc_lr=getFairGauge(fairRslt_roc_lr[3],0,2,flow=0.4,fhigh=0.63)

    #cpp
    fairRslt_cpp_lr = cpp_lr_df[sen_attr].values.tolist()
    gaugeSPD_cpp_lr=getFairGauge(fairRslt_cpp_lr[0],-1,1)
    gaugeEOD_cpp_lr=getFairGauge(fairRslt_cpp_lr[1],-1,1)
    gaugeAOD_cpp_lr=getFairGauge(fairRslt_cpp_lr[2],-1,1)
    gaugeDI_cpp_lr=getFairGauge(fairRslt_cpp_lr[3],0,2,flow=0.4,fhigh=0.63)


    return render_template(
        'fairness_lr.html',
        class_metrics_score=class_metrics_score,
        sen_attr=sen_attr,
        sen_attr1 = sen_attr1,
        sen_attr2 = sen_attr2,
        num_instances = num_instance,
        num_priv_instances = num_priv_instance,
        num_unpriv_instances = num_unpriv_instance,

        num_positives = num_positive,
        num_priv_positives = num_priv_positive,
        num_unpriv_positives = num_unpriv_positive,

        metric_attr=metric_attr,
        metric_score1=metric_score1,
        metric_score2=metric_score2,

        selection_rate = sel_rate,
        selection_priv_rate = sel_priv_rate,
        selection_unpriv_rate = sel_unpriv_rate,

        tpr = TPR,
        tprpriv = TPRPriv,
        tprunpriv = TPRUnpriv,

        fpr = FPR,
        fprpriv = FPRPriv,
        fprunpriv = FPRUnpriv,

        tnr = TNR,
        tnrpriv =TNRPriv,
        tnrunpriv = TNRUnpriv,
        
        fnr = FNR,
        fnrpriv = FNRPriv,
        fnrunpriv = FNRUnpriv,

        metric_fair = metric_fair,
        fairnesslvl = fairnesslvl,

        gaugeSPD = Markup(gaugeSPD.render_embed()),
        gaugeEOD = Markup(gaugeEOD.render_embed()),
        gaugeAOD = Markup(gaugeAOD.render_embed()),
        gaugeDI = Markup(gaugeDI.render_embed()),


        bal_acc_Scatter = Markup(bal_acc_Scatter.render_embed()),
        eq_opp_diff_Scatter = Markup(eq_opp_diff_Scatter.render_embed()),
        error_rate_diff_Scatter = Markup(error_rate_diff_Scatter.render_embed()),
        consistency_Scatter = Markup(consistency_Scatter.render_embed()),

        gaugeSPD_rw = Markup(gaugeSPD_rw.render_embed()),
        gaugeEOD_rw = Markup(gaugeEOD_rw.render_embed()),
        gaugeAOD_rw = Markup(gaugeAOD_rw.render_embed()),
        gaugeDI_rw = Markup(gaugeDI_rw.render_embed()),

        gaugeSPD_lfr = Markup(gaugeSPD_lfr.render_embed()),
        gaugeEOD_lfr = Markup(gaugeEOD_lfr.render_embed()),
        gaugeAOD_lfr = Markup(gaugeAOD_lfr.render_embed()),
        gaugeDI_lfr = Markup(gaugeDI_lfr.render_embed()),

        gaugeSPD_di = Markup(gaugeSPD_di.render_embed()),
        gaugeEOD_di = Markup(gaugeEOD_di.render_embed()),
        gaugeAOD_di = Markup(gaugeAOD_di.render_embed()),
        gaugeDI_di = Markup(gaugeDI_di.render_embed()),

        spd_Scatter = Markup(spd_Scatter.render_embed()),
        pr_eq_opp_diff_Scatter = Markup(pr_eq_opp_diff_Scatter.render_embed()),
        pr_avg_odds_diff_Scatter = Markup(pr_avg_odds_diff_Scatter.render_embed()),
        pr_di_Scatter = Markup(pr_di_Scatter.render_embed()),

        gaugeSPD_debias = Markup(gaugeSPD_debias.render_embed()),
        gaugeEOD_debias = Markup(gaugeEOD_debias.render_embed()),
        gaugeAOD_debias = Markup(gaugeAOD_debias.render_embed()),
        gaugeDI_debias = Markup(gaugeDI_debias.render_embed()),

        gaugeSPD_egr_lr = Markup(gaugeSPD_egr_lr.render_embed()),
        gaugeEOD_egr_lr = Markup(gaugeEOD_egr_lr.render_embed()),
        gaugeAOD_egr_lr = Markup(gaugeAOD_egr_lr.render_embed()),
        gaugeDI_egr_lr = Markup(gaugeDI_egr_lr.render_embed()),
    
        gaugeSPD_eqOdds_lr=Markup(gaugeSPD_eqOdds_lr.render_embed()),
        gaugeEOD_eqOdds_lr=Markup(gaugeEOD_eqOdds_lr.render_embed()),
        gaugeAOD_eqOdds_lr=Markup(gaugeAOD_eqOdds_lr.render_embed()),
        gaugeDI_eqOdds_lr=Markup(gaugeDI_eqOdds_lr.render_embed()),

        gaugeSPD_roc_lr = Markup(gaugeSPD_roc_lr.render_embed()),
        gaugeEOD_roc_lr = Markup(gaugeEOD_roc_lr.render_embed()),
        gaugeAOD_roc_lr = Markup(gaugeAOD_roc_lr.render_embed()),
        gaugeDI_roc_lr = Markup(gaugeDI_roc_lr.render_embed()),

        gaugeSPD_cpp_lr = Markup(gaugeSPD_cpp_lr.render_embed()),
        gaugeEOD_cpp_lr = Markup(gaugeEOD_cpp_lr.render_embed()),
        gaugeAOD_cpp_lr = Markup(gaugeAOD_cpp_lr.render_embed()),
        gaugeDI_cpp_lr = Markup(gaugeDI_cpp_lr.render_embed()),
        )


@app.route('/fairness_svc.html',methods=['GET','POST'])
def fairnessInvestigate_svc():
    lr_model, preprorcessor, X_train, y_train, X_test, y_test = loadDataAndModels('data/svcModel/svc_predictor.pkl')
    class_metrics_score=0
    metric_score1=0
    metric_score2=0
    sen_attr=''
    metric_attr=''
    sen_attr1=''
    sen_attr2=''
    metric_fair=''

    if request.method == 'POST':
        sen_attr = request.form.get('sensitiveAttr')
        if sen_attr=='gender':
            sen_attr1 = 'Male'
            sen_attr2 = 'Female'
        elif sen_attr =='race':
            sen_attr1 ='Caucasian'
            sen_attr2= 'Others'
        else:
            sen_attr1 = 'Hispanic'
            sen_attr2 = 'Others'

        metric_attr = request.form.get('perfMetrics')
        metric_fair = request.form.get('fairMetric')
    else:
        sen_attr ='gender'
        sen_attr1 = 'Male'
        sen_attr2 = 'Female'
        metric_attr = 'Accuracy'
        metric_fair = 'Consistency'
        

    dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups = constructAIF36ODataset(X_train, y_train, X_test, y_test, sen_attr)
    
    y_train_pred, y_test_pred = getPredictions(X_train,X_test,lr_model)
    _, metric_test = getClassMetricDataset(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups, y_train_pred, y_test_pred)

    fairRslt =getFairRslt(metric_test)

    # bal_acc,eq_opp_diff,error_rate_diff,consistency = getThresholdMetric(metric_test)

    # thresholdFairDf=pd.DataFrame({'bal_acc':np.round(bal_acc,4),'eq_opp_diff':np.round(eq_opp_diff,4),'error_rate_diff':np.round(error_rate_diff,4),'consistency':np.round(consistency,4),'thresh':np.round(np.linspace(0.01,0.99,100),3)})
    
    num_instance = metric_test.num_instances()
    num_priv_instance = metric_test.num_instances(privileged = True)
    num_unpriv_instance = metric_test.num_instances(privileged = False)

    num_positive = metric_test.num_positives()
    num_priv_positive = metric_test.num_positives(privileged = True)
    num_unpriv_positive = metric_test.num_positives(privileged = False)

    sel_rate = np.round(metric_test.selection_rate(),3)
    sel_priv_rate = np.round(metric_test.selection_rate(privileged = True), 3)
    sel_unpriv_rate = np.round(metric_test.selection_rate(privileged = False), 3)

    TPR = np.round(metric_test.true_positive_rate(),3)
    TPRPriv = np.round(metric_test.true_positive_rate(privileged = True),3)
    TPRUnpriv = np.round(metric_test.true_positive_rate(privileged = False), 3)

    TNR = np.round(metric_test.true_negative_rate(),3)
    TNRPriv = np.round(metric_test.true_negative_rate(privileged = True),3)
    TNRUnpriv = np.round(metric_test.true_negative_rate(privileged = False),3)

    FPR = np.round(metric_test.false_positive_rate(),3)
    FPRPriv = np.round(metric_test.false_positive_rate(privileged = True),3)
    FPRUnpriv = np.round(metric_test.false_positive_rate(privileged = False),3)

    FNR = np.round(metric_test.false_negative_rate(),3)
    FNRPriv = np.round(metric_test.false_negative_rate(privileged = True),3)
    FNRUnpriv = np.round(metric_test.false_negative_rate(privileged = False),3)

    fairnesslvl=getFairnessMetric(metric_test,metric_fair)

    if metric_attr=='Accuracy':
        class_metrics_score = np.round(metric_test.accuracy(),3)
        metric_score1 = np.round(metric_test.accuracy(privileged=True),3)
        metric_score2 = np.round(metric_test.accuracy(privileged=False),3)
    elif metric_attr =='Recall':
        class_metrics_score = np.round(metric_test.recall(),3)
        metric_score1 = np.round(metric_test.recall(privileged=True),3)
        metric_score2 = np.round(metric_test.recall(privileged=False),3)
    elif metric_attr =='Precision':
        class_metrics_score = np.round(metric_test.precision(),3)
        metric_score1 = np.round(metric_test.precision(privileged=True),3)
        metric_score2 = np.round(metric_test.precision(privileged=False),3)
    elif metric_attr =='F1-Score':
        class_metrics_score = np.round(f1_score(metric_test),3)
        metric_score1 = np.round(f1_score(metric_test,privileged=True),3)
        metric_score2 = np.round(f1_score(metric_test,privileged=False),3)
    else:
        class_metrics_score = np.round(balanced_accuracy(metric_test),3)
        metric_score1 = np.round(balanced_accuracy(metric_test, privileged=True),3)
        metric_score2 = np.round(balanced_accuracy(metric_test, privileged=False),3)

    gaugeSPD=getFairGauge(fairRslt[0],-1,1)
    gaugeEOD=getFairGauge(fairRslt[1],-1,1)
    gaugeAOD=getFairGauge(fairRslt[2],-1,1)
    gaugeDI=getFairGauge(fairRslt[3],0,2,flow=0.4,fhigh=0.63)

    thresholdFairDf = pd.read_csv(os.path.join('fairness','thresholdFairDf_'+sen_attr+'_svc.csv'))
    bal_acc_Scatter=getThresholdFair(thresholdFairDf, 'spd','Threshold','Statistical Parity Difference')
    eq_opp_diff_Scatter=getThresholdFair(thresholdFairDf,'eq_opp_diff','Threshold','eq_opp_diff')
    error_rate_diff_Scatter = getThresholdFair(thresholdFairDf, 'avg_odds_diff','Threshold','Average odds Difference')
    consistency_Scatter = getThresholdFair(thresholdFairDf, 'dispImp','Threshold','Disparate Impact')

    _,metric_test_svc=reweighing(unprivileged_groups,privileged_groups,dataset_orig_train,dataset_orig_test,'SVC')
    fairRslt_svc =getFairRslt(metric_test_svc)

    gaugeSPD_svc=getFairGauge(fairRslt_svc[0],-1,1)
    gaugeEOD_svc=getFairGauge(fairRslt_svc[1],-1,1)
    gaugeAOD_svc=getFairGauge(fairRslt_svc[2],-1,1)
    gaugeDI_svc=getFairGauge(fairRslt_svc[3],0,2,flow=0.4,fhigh=0.63)

    prThresholdFairDf = pd.read_csv(os.path.join('fairness','prDf_'+sen_attr+'.csv'))
    spd_Scatter = getThresholdFair(prThresholdFairDf, 'Statistical Parity Difference','threshold','Statistical Parity Difference')
    pr_eq_opp_diff_Scatter = getThresholdFair(prThresholdFairDf,'Equal Opportunity Difference','threshold','Equal Opportunity Difference')
    pr_avg_odds_diff_Scatter = getThresholdFair(prThresholdFairDf, 'Average Odds Difference','threshold','Average Odds Difference')
    pr_di_Scatter = getThresholdFair(prThresholdFairDf, 'Disparat Impact','threshold','Disparat Impact')
    
    metric_test_eqOdds_svc= eqOddsPostProcessing(unprivileged_groups,privileged_groups,dataset_orig_test,dataset_orig_test,'SVC')
    fairRslt_eqOdds_svc = getFairRslt(metric_test_eqOdds_svc)

    gaugeSPD_eqOdds_svc=getFairGauge(fairRslt_eqOdds_svc[0],-1,1)
    gaugeEOD_eqOdds_svc=getFairGauge(fairRslt_eqOdds_svc[1],-1,1)
    gaugeAOD_eqOdds_svc=getFairGauge(fairRslt_eqOdds_svc[2],-1,1)
    gaugeDI_eqOdds_svc=getFairGauge(fairRslt_eqOdds_svc[3],0,2,flow=0.4,fhigh=0.63)

    # LFR
    svc_lfr = pickle.load(open('fairness/'+sen_attr+'_svc_lfr.pkl','rb'))
    y_test_transf_pred = svc_lfr.predict(dataset_orig_test.features)
    y_train_transf_pred = svc_lfr.predict(dataset_orig_train.features)
    _,metric_test_lfr_svc=getClassMetricDataset(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups, y_train_transf_pred, y_test_transf_pred)

    fairRslt_lfr_svc =getFairRslt(metric_test_lfr_svc)

    gaugeSPD_lfr_svc=getFairGauge(fairRslt_lfr_svc[0],-1,1)
    gaugeEOD_lfr_svc=getFairGauge(fairRslt_lfr_svc[1],-1,1)
    gaugeAOD_lfr_svc=getFairGauge(fairRslt_lfr_svc[2],-1,1)
    gaugeDI_lfr_svc=getFairGauge(fairRslt_lfr_svc[3],0,2,flow=0.4,fhigh=0.63)

    # Disparate Remover
    _,metric_test_di_svc = disparateImpactRemover(unprivileged_groups,privileged_groups,dataset_orig_train,dataset_orig_test,'SVC',repair_level=1.0)
    fairRslt_di_svc =getFairRslt(metric_test_di_svc)

    gaugeSPD_di_svc=getFairGauge(fairRslt_di_svc[0],-1,1)
    gaugeEOD_di_svc=getFairGauge(fairRslt_di_svc[1],-1,1)
    gaugeAOD_di_svc=getFairGauge(fairRslt_di_svc[2],-1,1)
    gaugeDI_di_svc=getFairGauge(fairRslt_di_svc[3],0,2,flow=0.4,fhigh=0.63)

    # Debiasing
    fairRslt_debias = debiased_df[sen_attr].values.tolist()
    gaugeSPD_debias=getFairGauge(fairRslt_debias[0],-1,1)
    gaugeEOD_debias=getFairGauge(fairRslt_debias[1],-1,1)
    gaugeAOD_debias=getFairGauge(fairRslt_debias[2],-1,1)
    gaugeDI_debias=getFairGauge(fairRslt_debias[3],0,2,flow=0.4,fhigh=0.63)

      # roc
    fairRslt_roc_svc = roc_svc_df[sen_attr].values.tolist()
    gaugeSPD_roc_svc=getFairGauge(fairRslt_roc_svc[0],-1,1)
    gaugeEOD_roc_svc=getFairGauge(fairRslt_roc_svc[1],-1,1)
    gaugeAOD_roc_svc=getFairGauge(fairRslt_roc_svc[2],-1,1)
    gaugeDI_roc_svc=getFairGauge(fairRslt_roc_svc[3],0,2,flow=0.4,fhigh=0.63)

    #cpp
    fairRslt_cpp_svc = cpp_svc_df[sen_attr].values.tolist()
    gaugeSPD_cpp_svc=getFairGauge(fairRslt_cpp_svc[0],-1,1)
    gaugeEOD_cpp_svc=getFairGauge(fairRslt_cpp_svc[1],-1,1)
    gaugeAOD_cpp_svc=getFairGauge(fairRslt_cpp_svc[2],-1,1)
    gaugeDI_cpp_svc=getFairGauge(fairRslt_cpp_svc[3],0,2,flow=0.4,fhigh=0.63)



    return render_template(
        'fairness_svc.html',
        class_metrics_score=class_metrics_score,
        sen_attr=sen_attr,
        sen_attr1 = sen_attr1,
        sen_attr2 = sen_attr2,
        num_instances = num_instance,
        num_priv_instances = num_priv_instance,
        num_unpriv_instances = num_unpriv_instance,

        num_positives = num_positive,
        num_priv_positives = num_priv_positive,
        num_unpriv_positives = num_unpriv_positive,

        metric_attr=metric_attr,
        metric_score1=metric_score1,
        metric_score2=metric_score2,

        selection_rate = sel_rate,
        selection_priv_rate = sel_priv_rate,
        selection_unpriv_rate = sel_unpriv_rate,

        tpr = TPR,
        tprpriv = TPRPriv,
        tprunpriv = TPRUnpriv,

        fpr = FPR,
        fprpriv = FPRPriv,
        fprunpriv = FPRUnpriv,

        tnr = TNR,
        tnrpriv =TNRPriv,
        tnrunpriv = TNRUnpriv,
        
        fnr = FNR,
        fnrpriv = FNRPriv,
        fnrunpriv = FNRUnpriv,

        metric_fair = metric_fair,
        fairnesslvl = fairnesslvl,

        gaugeSPD = Markup(gaugeSPD.render_embed()),
        gaugeEOD = Markup(gaugeEOD.render_embed()),
        gaugeAOD = Markup(gaugeAOD.render_embed()),
        gaugeDI = Markup(gaugeDI.render_embed()),

        bal_acc_Scatter = Markup(bal_acc_Scatter.render_embed()),
        eq_opp_diff_Scatter = Markup(eq_opp_diff_Scatter.render_embed()),
        error_rate_diff_Scatter = Markup(error_rate_diff_Scatter.render_embed()),
        consistency_Scatter = Markup(consistency_Scatter.render_embed()),

        gaugeSPD_svc = Markup(gaugeSPD_svc.render_embed()),
        gaugeEOD_svc = Markup(gaugeEOD_svc.render_embed()),
        gaugeAOD_svc = Markup(gaugeAOD_svc.render_embed()),
        gaugeDI_svc = Markup(gaugeDI_svc.render_embed()),

        gaugeSPD_lfr_svc=Markup(gaugeSPD_lfr_svc.render_embed()),
        gaugeEOD_lfr_svc=Markup(gaugeEOD_lfr_svc.render_embed()),
        gaugeAOD_lfr_svc=Markup(gaugeAOD_lfr_svc.render_embed()),
        gaugeDI_lfr_svc=Markup(gaugeDI_lfr_svc.render_embed()),

        gaugeSPD_di_svc=Markup(gaugeSPD_di_svc.render_embed()),
        gaugeEOD_di_svc=Markup(gaugeEOD_di_svc.render_embed()),
        gaugeAOD_di_svc=Markup(gaugeAOD_di_svc.render_embed()),
        gaugeDI_di_svc=Markup(gaugeDI_di_svc.render_embed()),
        
        spd_Scatter = Markup(spd_Scatter.render_embed()),
        pr_eq_opp_diff_Scatter = Markup(pr_eq_opp_diff_Scatter.render_embed()),
        pr_avg_odds_diff_Scatter = Markup(pr_avg_odds_diff_Scatter.render_embed()),
        pr_di_Scatter = Markup(pr_di_Scatter.render_embed()),
        
        gaugeSPD_debias = Markup(gaugeSPD_debias.render_embed()),
        gaugeEOD_debias = Markup(gaugeEOD_debias.render_embed()),
        gaugeAOD_debias = Markup(gaugeAOD_debias.render_embed()),
        gaugeDI_debias = Markup(gaugeDI_debias.render_embed()),

        gaugeSPD_eqOdds_svc=Markup(gaugeSPD_eqOdds_svc.render_embed()),
        gaugeEOD_eqOdds_svc=Markup(gaugeEOD_eqOdds_svc.render_embed()),
        gaugeAOD_eqOdds_svc=Markup(gaugeAOD_eqOdds_svc.render_embed()),
        gaugeDI_eqOdds_svc=Markup(gaugeDI_eqOdds_svc.render_embed()),

        gaugeSPD_roc_svc = Markup(gaugeSPD_roc_svc.render_embed()),
        gaugeEOD_roc_svc = Markup(gaugeEOD_roc_svc.render_embed()),
        gaugeAOD_roc_svc = Markup(gaugeAOD_roc_svc.render_embed()),
        gaugeDI_roc_svc = Markup(gaugeDI_roc_svc.render_embed()),

        gaugeSPD_cpp_svc = Markup(gaugeSPD_cpp_svc.render_embed()),
        gaugeEOD_cpp_svc = Markup(gaugeEOD_cpp_svc.render_embed()),
        gaugeAOD_cpp_svc = Markup(gaugeAOD_cpp_svc.render_embed()),
        gaugeDI_cpp_svc = Markup(gaugeDI_cpp_svc.render_embed()),
        )

@app.route('/fairness_dt.html',methods=['GET','POST'])
def fairnessInvestigate_dt():
    lr_model, preprorcessor, X_train, y_train, X_test, y_test = loadDataAndModels('data/dtModel/dt_predictor.pkl')
    class_metrics_score=0
    metric_score1=0
    metric_score2=0
    sen_attr=''
    metric_attr=''
    sen_attr1=''
    sen_attr2=''
    metric_fair=''

    if request.method == 'POST':
        sen_attr = request.form.get('sensitiveAttr')
        if sen_attr=='gender':
            sen_attr1 = 'Male'
            sen_attr2 = 'Female'
        elif sen_attr =='race':
            sen_attr1 ='Caucasian'
            sen_attr2= 'Others'
        else:
            sen_attr1 = 'Hispanic'
            sen_attr2 = 'Others'

        metric_attr = request.form.get('perfMetrics')
        metric_fair = request.form.get('fairMetric')
    else:
        sen_attr ='gender'
        sen_attr1 = 'Male'
        sen_attr2 = 'Female'
        metric_attr = 'Accuracy'
        metric_fair = 'Consistency'
        
    dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups = constructAIF36ODataset(X_train, y_train, X_test, y_test, sen_attr)
    
    y_train_pred, y_test_pred = getPredictions(X_train,X_test,lr_model)
    _, metric_test = getClassMetricDataset(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups, y_train_pred, y_test_pred)

    fairRslt =getFairRslt(metric_test)

    # bal_acc,eq_opp_diff,error_rate_diff,consistency = getThresholdMetric(metric_test)
    # thresholdFairDf=pd.DataFrame({'bal_acc':np.round(bal_acc,4),'eq_opp_diff':np.round(eq_opp_diff,4),'error_rate_diff':np.round(error_rate_diff,4),'consistency':np.round(consistency,4),'thresh':np.round(np.linspace(0.01,0.99,100),3)})
    
    num_instance = metric_test.num_instances()
    num_priv_instance = metric_test.num_instances(privileged = True)
    num_unpriv_instance = metric_test.num_instances(privileged = False)

    num_positive = metric_test.num_positives()
    num_priv_positive = metric_test.num_positives(privileged = True)
    num_unpriv_positive = metric_test.num_positives(privileged = False)

    sel_rate = np.round(metric_test.selection_rate(),3)
    sel_priv_rate = np.round(metric_test.selection_rate(privileged = True), 3)
    sel_unpriv_rate = np.round(metric_test.selection_rate(privileged = False), 3)

    TPR = np.round(metric_test.true_positive_rate(),3)
    TPRPriv = np.round(metric_test.true_positive_rate(privileged = True),3)
    TPRUnpriv = np.round(metric_test.true_positive_rate(privileged = False), 3)

    TNR = np.round(metric_test.true_negative_rate(),3)
    TNRPriv = np.round(metric_test.true_negative_rate(privileged = True),3)
    TNRUnpriv = np.round(metric_test.true_negative_rate(privileged = False),3)

    FPR = np.round(metric_test.false_positive_rate(),3)
    FPRPriv = np.round(metric_test.false_positive_rate(privileged = True),3)
    FPRUnpriv = np.round(metric_test.false_positive_rate(privileged = False),3)

    FNR = np.round(metric_test.false_negative_rate(),3)
    FNRPriv = np.round(metric_test.false_negative_rate(privileged = True),3)
    FNRUnpriv = np.round(metric_test.false_negative_rate(privileged = False),3)

    fairnesslvl=getFairnessMetric(metric_test,metric_fair)

    if metric_attr=='Accuracy':
        class_metrics_score = np.round(metric_test.accuracy(),3)
        metric_score1 = np.round(metric_test.accuracy(privileged=True),3)
        metric_score2 = np.round(metric_test.accuracy(privileged=False),3)
    elif metric_attr =='Recall':
        class_metrics_score = np.round(metric_test.recall(),3)
        metric_score1 = np.round(metric_test.recall(privileged=True),3)
        metric_score2 = np.round(metric_test.recall(privileged=False),3)
    elif metric_attr =='Precision':
        class_metrics_score = np.round(metric_test.precision(),3)
        metric_score1 = np.round(metric_test.precision(privileged=True),3)
        metric_score2 = np.round(metric_test.precision(privileged=False),3)
    elif metric_attr =='F1-Score':
        class_metrics_score = np.round(f1_score(metric_test),3)
        metric_score1 = np.round(f1_score(metric_test,privileged=True),3)
        metric_score2 = np.round(f1_score(metric_test,privileged=False),3)
    else:
        class_metrics_score = np.round(balanced_accuracy(metric_test),3)
        metric_score1 = np.round(balanced_accuracy(metric_test, privileged=True),3)
        metric_score2 = np.round(balanced_accuracy(metric_test, privileged=False),3)

    gaugeSPD=getFairGauge(fairRslt[0],-1,1)
    gaugeEOD=getFairGauge(fairRslt[1],-1,1)
    gaugeAOD=getFairGauge(fairRslt[2],-1,1)
    gaugeDI=getFairGauge(fairRslt[3],0,2,flow=0.4,fhigh=0.63)

    thresholdFairDf = pd.read_csv(os.path.join('fairness','thresholdFairDf_'+sen_attr+'_dt.csv'))
    bal_acc_Scatter=getThresholdFair(thresholdFairDf, 'spd','Threshold','Statistical Parity Difference')
    eq_opp_diff_Scatter=getThresholdFair(thresholdFairDf,'eq_opp_diff','Threshold','eq_opp_diff')
    error_rate_diff_Scatter = getThresholdFair(thresholdFairDf, 'avg_odds_diff','Threshold','Average odds Difference')
    consistency_Scatter = getThresholdFair(thresholdFairDf, 'dispImp','Threshold','Disparate Impact')

    _,metric_test_dt=reweighing(unprivileged_groups,privileged_groups,dataset_orig_train,dataset_orig_test,'Decision Tree')
    fairRslt_dt =getFairRslt(metric_test_dt)

    gaugeSPD_dt=getFairGauge(fairRslt_dt[0], -1, 1)
    gaugeEOD_dt=getFairGauge(fairRslt_dt[1], -1, 1)
    gaugeAOD_dt=getFairGauge(fairRslt_dt[2], -1, 1)
    gaugeDI_dt=getFairGauge(fairRslt_dt[3], 0, 2, flow=0.4, fhigh=0.63)

    prThresholdFairDf = pd.read_csv(os.path.join('fairness','prDf_'+sen_attr+'.csv'))
    spd_Scatter = getThresholdFair(prThresholdFairDf, 'Statistical Parity Difference','threshold','Statistical Parity Difference')
    pr_eq_opp_diff_Scatter = getThresholdFair(prThresholdFairDf,'Equal Opportunity Difference','threshold','Equal Opportunity Difference')
    pr_avg_odds_diff_Scatter = getThresholdFair(prThresholdFairDf, 'Average Odds Difference','threshold','Average Odds Difference')
    pr_di_Scatter = getThresholdFair(prThresholdFairDf, 'Disparat Impact','threshold','Disparat Impact')

    metric_test_eqOdds_dt= eqOddsPostProcessing(unprivileged_groups,privileged_groups,dataset_orig_test,dataset_orig_test,'Decision Tree')
    fairRslt_eqOdds_dt = getFairRslt(metric_test_eqOdds_dt)

    gaugeSPD_eqOdds_dt=getFairGauge(fairRslt_eqOdds_dt[0],-1, 1)
    gaugeEOD_eqOdds_dt=getFairGauge(fairRslt_eqOdds_dt[1],-1, 1)
    gaugeAOD_eqOdds_dt=getFairGauge(fairRslt_eqOdds_dt[2],-1, 1)
    gaugeDI_eqOdds_dt=getFairGauge(fairRslt_eqOdds_dt[3], 0, 2, flow=0.4, fhigh=0.63)

    # LFR
    dt_lfr = pickle.load(open('fairness/'+sen_attr+'_dt_lfr.pkl','rb'))
    y_test_transf_pred = dt_lfr.predict(dataset_orig_test.features)
    y_train_transf_pred = dt_lfr.predict(dataset_orig_train.features)
    _,metric_test_lfr_dt=getClassMetricDataset(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups, y_train_transf_pred, y_test_transf_pred)

    fairRslt_lfr_dt =getFairRslt(metric_test_lfr_dt)

    gaugeSPD_lfr_dt=getFairGauge(fairRslt_lfr_dt[0],-1,1)
    gaugeEOD_lfr_dt=getFairGauge(fairRslt_lfr_dt[1],-1,1)
    gaugeAOD_lfr_dt=getFairGauge(fairRslt_lfr_dt[2],-1,1)
    gaugeDI_lfr_dt=getFairGauge(fairRslt_lfr_dt[3],0,2,flow=0.4,fhigh=0.63)

    # Disparate Remover
    _,metric_test_di_dt = disparateImpactRemover(unprivileged_groups,privileged_groups,dataset_orig_train,dataset_orig_test,'Decision Tree',repair_level=1.0)
    fairRslt_di_dt =getFairRslt(metric_test_di_dt)

    gaugeSPD_di_dt=getFairGauge(fairRslt_di_dt[0],-1,1)
    gaugeEOD_di_dt=getFairGauge(fairRslt_di_dt[1],-1,1)
    gaugeAOD_di_dt=getFairGauge(fairRslt_di_dt[2],-1,1)
    gaugeDI_di_dt=getFairGauge(fairRslt_di_dt[3],0,2,flow=0.4,fhigh=0.63)

    # Debiasing
    fairRslt_debias = debiased_df[sen_attr].values.tolist()
    gaugeSPD_debias=getFairGauge(fairRslt_debias[0],-1,1)
    gaugeEOD_debias=getFairGauge(fairRslt_debias[1],-1,1)
    gaugeAOD_debias=getFairGauge(fairRslt_debias[2],-1,1)
    gaugeDI_debias=getFairGauge(fairRslt_debias[3],0,2,flow=0.4,fhigh=0.63)

    # exp_grad_red
    fairRslt_egr_dt = exp_grad_red_dt[sen_attr].values.tolist()
    gaugeSPD_egr_dt=getFairGauge(fairRslt_egr_dt[0],-1,1)
    gaugeEOD_egr_dt=getFairGauge(fairRslt_egr_dt[1],-1,1)
    gaugeAOD_egr_dt=getFairGauge(fairRslt_egr_dt[2],-1,1)
    gaugeDI_egr_dt=getFairGauge(fairRslt_egr_dt[3],0,2,flow=0.4,fhigh=0.63)

    # roc
    fairRslt_roc_dt = roc_dt_df[sen_attr].values.tolist()
    gaugeSPD_roc_dt=getFairGauge(fairRslt_roc_dt[0],-1,1)
    gaugeEOD_roc_dt=getFairGauge(fairRslt_roc_dt[1],-1,1)
    gaugeAOD_roc_dt=getFairGauge(fairRslt_roc_dt[2],-1,1)
    gaugeDI_roc_dt=getFairGauge(fairRslt_roc_dt[3],0,2,flow=0.4,fhigh=0.63)

    #cpp
    fairRslt_cpp_dt = cpp_dt_df[sen_attr].values.tolist()
    gaugeSPD_cpp_dt=getFairGauge(fairRslt_cpp_dt[0],-1,1)
    gaugeEOD_cpp_dt=getFairGauge(fairRslt_cpp_dt[1],-1,1)
    gaugeAOD_cpp_dt=getFairGauge(fairRslt_cpp_dt[2],-1,1)
    gaugeDI_cpp_dt=getFairGauge(fairRslt_cpp_dt[3],0,2,flow=0.4,fhigh=0.63)

    return render_template(
        'fairness_dt.html',
        class_metrics_score=class_metrics_score,
        sen_attr=sen_attr,
        sen_attr1 = sen_attr1,
        sen_attr2 = sen_attr2,
        num_instances = num_instance,
        num_priv_instances = num_priv_instance,
        num_unpriv_instances = num_unpriv_instance,

        num_positives = num_positive,
        num_priv_positives = num_priv_positive,
        num_unpriv_positives = num_unpriv_positive,

        metric_attr=metric_attr,
        metric_score1=metric_score1,
        metric_score2=metric_score2,

        selection_rate = sel_rate,
        selection_priv_rate = sel_priv_rate,
        selection_unpriv_rate = sel_unpriv_rate,

        tpr = TPR,
        tprpriv = TPRPriv,
        tprunpriv = TPRUnpriv,

        fpr = FPR,
        fprpriv = FPRPriv,
        fprunpriv = FPRUnpriv,

        tnr = TNR,
        tnrpriv =TNRPriv,

        tnrunpriv = TNRUnpriv,
        
        fnr = FNR,
        fnrpriv = FNRPriv,
        fnrunpriv = FNRUnpriv,

        metric_fair = metric_fair,
        fairnesslvl = fairnesslvl,

        gaugeSPD = Markup(gaugeSPD.render_embed()),
        gaugeEOD = Markup(gaugeEOD.render_embed()),
        gaugeAOD = Markup(gaugeAOD.render_embed()),
        gaugeDI = Markup(gaugeDI.render_embed()),

        gaugeSPD_lfr_dt=Markup(gaugeSPD_lfr_dt.render_embed()),
        gaugeEOD_lfr_dt=Markup(gaugeEOD_lfr_dt.render_embed()),
        gaugeAOD_lfr_dt=Markup(gaugeAOD_lfr_dt.render_embed()),
        gaugeDI_lfr_dt=Markup(gaugeDI_lfr_dt.render_embed()),

        gaugeSPD_di_dt=Markup(gaugeSPD_di_dt.render_embed()),
        gaugeEOD_di_dt=Markup(gaugeEOD_di_dt.render_embed()),
        gaugeAOD_di_dt=Markup(gaugeAOD_di_dt.render_embed()),
        gaugeDI_di_dt=Markup(gaugeDI_di_dt.render_embed()),

        bal_acc_Scatter = Markup(bal_acc_Scatter.render_embed()),
        eq_opp_diff_Scatter = Markup(eq_opp_diff_Scatter.render_embed()),
        error_rate_diff_Scatter = Markup(error_rate_diff_Scatter.render_embed()),
        consistency_Scatter = Markup(consistency_Scatter.render_embed()),

        gaugeSPD_dt = Markup(gaugeSPD_dt.render_embed()),
        gaugeEOD_dt = Markup(gaugeEOD_dt.render_embed()),
        gaugeAOD_dt = Markup(gaugeAOD_dt.render_embed()),
        gaugeDI_dt = Markup(gaugeDI_dt.render_embed()),

        spd_Scatter = Markup(spd_Scatter.render_embed()),
        pr_eq_opp_diff_Scatter = Markup(pr_eq_opp_diff_Scatter.render_embed()),
        pr_avg_odds_diff_Scatter = Markup(pr_avg_odds_diff_Scatter.render_embed()),
        pr_di_Scatter = Markup(pr_di_Scatter.render_embed()),
        
        gaugeSPD_debias = Markup(gaugeSPD_debias.render_embed()),
        gaugeEOD_debias = Markup(gaugeEOD_debias.render_embed()),
        gaugeAOD_debias = Markup(gaugeAOD_debias.render_embed()),
        gaugeDI_debias = Markup(gaugeDI_debias.render_embed()),

        gaugeSPD_egr_dt = Markup(gaugeSPD_egr_dt.render_embed()),
        gaugeEOD_egr_dt = Markup(gaugeEOD_egr_dt.render_embed()),
        gaugeAOD_egr_dt = Markup(gaugeAOD_egr_dt.render_embed()),
        gaugeDI_egr_dt = Markup(gaugeDI_egr_dt.render_embed()),

        gaugeSPD_eqOdds_dt=Markup(gaugeSPD_eqOdds_dt.render_embed()),
        gaugeEOD_eqOdds_dt=Markup(gaugeEOD_eqOdds_dt.render_embed()),
        gaugeAOD_eqOdds_dt=Markup(gaugeAOD_eqOdds_dt.render_embed()),
        gaugeDI_eqOdds_dt=Markup(gaugeDI_eqOdds_dt.render_embed()),

        gaugeSPD_roc_dt = Markup(gaugeSPD_roc_dt.render_embed()),
        gaugeEOD_roc_dt = Markup(gaugeEOD_roc_dt.render_embed()),
        gaugeAOD_roc_dt= Markup(gaugeAOD_roc_dt.render_embed()),
        gaugeDI_roc_dt= Markup(gaugeDI_roc_dt.render_embed()),

        gaugeSPD_cpp_dt = Markup(gaugeSPD_cpp_dt.render_embed()),
        gaugeEOD_cpp_dt = Markup(gaugeEOD_cpp_dt.render_embed()),
        gaugeAOD_cpp_dt = Markup(gaugeAOD_cpp_dt.render_embed()),
        gaugeDI_cpp_dt = Markup(gaugeDI_cpp_dt.render_embed()),
        )

if __name__ == "__main__":

    print('sklearn.version',sklearn.__version__)
    app.run(
        # host='0.0.0.0',
        port=5001,
        debug = True)