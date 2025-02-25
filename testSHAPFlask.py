from turtle import get_shapepoly
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.DataProcessing import *
import shap

from flask import Flask, render_template

from utils.mlVisual import *
from ml.nearestNeighbor import *
from utils.modelMetricVisualization import *
from ml.modelExplainer import *
from lime import lime_tabular



model_lr = pickle.load(open('data/lrModel/lr_predictor.pkl','rb'))
data_path_lr='data/lrModel/studDf.csv'
model_svc = pickle.load(open('data/svcModel/svc_predictor.pkl','rb'))
data_path_svc='data/svcModel/studDf.csv'
model_dt = pickle.load(open('data/dtModel/dt_predictor.pkl','rb'))
data_path_dt='data/dtModel/studDf.csv'


app =  Flask(__name__)

@app.route('/testFlaskSHAP.html')
def display():
    exp_html=getExpHtml(model_lr['classification'], 8258207, data_path_lr)
    return render_template('testFlaskSHAP.html',exp=exp_html)


@app.route('/shapPlot.html')
def displayShap():
    shap_values,explainerShap,neighbors,cols=getModelExplainerShapSVC(model_dt['classification'], 8258207,data_path_dt)
    shap_plots = {}

    for i in range(len(shap_values)):
        ind = i
        shap_plots[i] = svc_force_plot_html(shap_values, explainerShap, ind, cols, str(neighbors[i]))

   
    return render_template('shapPlot.html', shap_plots = shap_plots)

if __name__=='__main__':
    app.run(debug=True, port=5003)





