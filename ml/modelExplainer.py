import pickle

import matplotlib
# from torch import ge
from ml.DataProcessing import *
# from DataProcessing import *
from ml.nearestNeighbor import *
# from nearestNeighbor import *
import shap
from lime import lime_tabular
import pandas as pd
import numpy as np
import sklearn

print('sklearn.version',sklearn.__version__)
aggDataDf = pd.read_csv('data/preprocessData/aggDataDf.csv')
X_train = pd.read_csv('data/preprocessData/X_train.csv')
X_test = pd.read_csv('data/preprocessData/X_test.csv')
y_train = pd.read_csv('data/preprocessData/y_train.csv')
preprocessor = pickle.load(open('data/preprocessData/preprocessing.pkl','rb'))



# preprocessor = preprocessing

X_train = preprocessor.fit_transform(X_train)
# preprocessor = preprocessor.fit(X_train)
# X_train = preprocessor.transform(X_train)
X_test = preprocessor.transform(X_test)



def getModelExplainerLIME():
    target_names=['Fail','Pass']
    cols =[col for col in aggDataDf.columns.to_list() if col not in ['useraccount_id','performance','labels']]
    explainer = lime_tabular.LimeTabularExplainer(X_train, mode="classification",
                                                class_names=target_names,
                                                feature_names=cols,
                                                )
    return explainer,cols

def getExpHtml(model, userId, data):
    explainer,cols = getModelExplainerLIME()
    _,idx,_,_=getNeigborsTable(userId,6,data)
    X_test_examining = X_test[idx]
    exp = explainer.explain_instance(
            X_test_examining,
            model.predict_proba,
            num_features=len(cols)
            )
    return exp.as_html()


#######################################################################################
### Shap Explainer
#######################################################################################

def getModelExplainerShap(model,userId,datapath):
    cols =[col for col in aggDataDf.columns.to_list() if col not in ['useraccount_id','performance','labels']]
    _, _, idx, users = getNeigborsTable(userId,6,datapath)
    indCurr=users.tolist().index(userId)

    neighbors = [user for user in users if user!=userId]

    explainerShap = shap.Explainer(
        model,
        X_train,
        feature_names=cols
    )
    shap_values = explainerShap(X_test[idx])
    shap_values = np.delete(shap_values.values,[indCurr],axis=0)

    # return shap_values,neighbors
    return shap_values,explainerShap,neighbors,cols

def force_plot_html(shap_values,explainer,ind, cols, user):
    
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_values[ind],
        out_names=user,
        feature_names=cols,
        matplotlib=False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return shap_html



def getModelExplainerShapSVCDT(model,userId,datapath):
    cols =[col for col in aggDataDf.columns.to_list() if col not in ['useraccount_id','performance','labels']]
    _, _, idx, users = getNeigborsTable(userId,6,datapath)
    indCurr=users.tolist().index(userId)

    neighbors = [user for user in users if user!=userId]
    X_train_summary = shap.kmeans(X_train,10)

    explainerShap = shap.KernelExplainer(
        model.predict_proba,
        X_train_summary,
        feature_names=cols
    )
    shap_values = explainerShap.shap_values(X_test[idx])
    shap_values = np.delete(shap_values[0],[indCurr],axis=0)

    # return shap_values,neighbors
    return shap_values,explainerShap,neighbors,cols

def svc_dt_force_plot_html(shap_values,explainer,ind, cols, user):
    
    force_plot = shap.force_plot(
        explainer.expected_value[0],
        shap_values[ind],
        out_names=user,
        feature_names=cols,
        matplotlib=False)
    shap_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    return shap_html