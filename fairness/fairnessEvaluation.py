from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric

import pickle
import numpy as np
import pandas as pd
import copy

def loadDataAndModels(modelPath):

    lr_model = pickle.load(open(modelPath,'rb'))
    preprocessor = pickle.load(open('data/preprocessData/preprocessing.pkl','rb'))

    X_train = pd.read_csv('data/preprocessData/X_train.csv')
    y_train = pd.read_csv('data/preprocessData/y_train.csv')

    X_test = pd.read_csv('data/preprocessData/X_test.csv')
    y_test = pd.read_csv('data/preprocessData/y_test.csv')

    return lr_model, preprocessor, X_train, y_train, X_test, y_test

def getDataset(loadDataAndModel,modelPath,attr):
    lr_model, preprocessor, X_train, y_train, X_test, y_test=loadDataAndModel(modelPath)

    train_df = pd.concat([X_train,y_train],axis=1)
    test_df = pd.concat([X_test,y_test],axis=1)

    dataset_orig_train = StandardDataset(
    df = train_df,
        label_name= 'labels',
        favorable_classes= [1],
        protected_attribute_names = [attr],
        privileged_classes = [np.array([1])]
    )

    dataset_orig_test = StandardDataset(
        df = test_df,
        label_name= 'labels',
        favorable_classes= [1],
        protected_attribute_names= [attr],
        privileged_classes = [np.array([1])]
    )
    unprivileged_groups=[{attr:0.0}]
    privileged_groups=[{attr:1.0}]
    return dataset_orig_train, dataset_orig_test,unprivileged_groups,privileged_groups,lr_model


# 生成aif360型数据集
def constructAIF36ODataset(X_train, y_train, X_test, y_test, protected_attr):
    train_df = pd.concat([X_train,y_train],axis=1)
    test_df = pd.concat([X_test,y_test],axis=1)

    dataset_orig_train = StandardDataset(
    df = train_df,
    label_name= 'labels',
    favorable_classes= [1],
    # protected_attribute_names = ['gender'],
    protected_attribute_names = [protected_attr],
    privileged_classes = [np.array([1])]
    )

    dataset_orig_test = StandardDataset(
        df = test_df,
        label_name= 'labels',
        favorable_classes= [1],
        # protected_attribute_names= ['gender'],
        protected_attribute_names = [protected_attr],
        privileged_classes = [np.array([1])]
    )

    unprivileged_groups=[{protected_attr:0.0}]
    privileged_groups=[{protected_attr:1.0}]

    return dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups

# 根据threshold生成 y_train_pred 和 y_test_pred
def getPredictions(X_train,X_test,model,threshold=None):
    
    y_train_pred_probas = model.predict_proba(X_train)
    y_test_pred_probas = model.predict_proba(X_test)
    pos_ind = 1
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if threshold:
        y_train_pred = (y_train_pred_probas[:,pos_ind]>=threshold).astype(np.float64)
        y_test_pred = (y_test_pred_probas[:,pos_ind]>=threshold).astype(np.float64)
    
    return y_train_pred, y_test_pred


def getPredictionsNonProbas(dataset_train,dataset_test,model,threshold=None):
    y_train_pred_probas = model.predict(dataset_train).scores
    y_test_pred_probas = model.predict(dataset_test).scores
    
    y_train_pred = model.predict(dataset_train)
    y_test_pred = model.predict(dataset_test)

    if threshold:
        y_train_pred = (y_train_pred_probas[:,0]>=threshold).astype(np.float64)
        y_test_pred = (y_test_pred_probas[:,0]>=threshold).astype(np.float64)
    
    return y_train_pred, y_test_pred


# 生成metric_train, metric_test
def getClassMetricDataset(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups, y_train_pred, y_test_pred):
    dataset_orig_train_pred = dataset_orig_train.copy()
    dataset_orig_train_pred.labels = y_train_pred.reshape(-1,1)

    dataset_orig_test_pred = dataset_orig_test.copy()
    dataset_orig_test_pred.labels = y_test_pred.reshape(-1,1)

    metric_train = ClassificationMetric(
        dataset_orig_train,
        dataset_orig_train_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    metric_test = ClassificationMetric(
        dataset_orig_test,
        dataset_orig_test_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    return metric_train, metric_test

def balanced_accuracy(metric_test,privileged=None):
    bal_acc =  (metric_test.true_positive_rate()+metric_test.true_negative_rate())/2
    if privileged:
        bal_acc = (metric_test.true_positive_rate(privileged=True)+metric_test.true_negative_rate(privileged=True))/2
    else:
        bal_acc = (metric_test.true_positive_rate(privileged=False)+metric_test.true_negative_rate(privileged=False))/2
    return bal_acc

def f1_score(metric_test,privileged=None):
    f1Score = 2*(metric_test.precision()*metric_test.recall())/(metric_test.precision()+metric_test.recall())
    if privileged:
        f1Score = 2*(metric_test.precision(privileged=True)*metric_test.recall(privileged=True))/(metric_test.precision(privileged=True)+metric_test.recall(privileged=True))
    else:
        f1Score = 2*(metric_test.precision(privileged=False)*metric_test.recall(privileged=False))/(metric_test.precision(privileged=False)+metric_test.recall(privileged=False))
    return f1Score



# lr_model, preprocessor, X_train, y_train, X_test, y_test = loadDataAndModels()
# dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups = constructAIF36ODataset(X_train, y_train, X_test, y_test, "gender")
# y_train_pred, y_test_pred = getPredictions(X_train,X_test,lr_model)
# metric_train, metric_test = getClassMetricDataset(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups, y_train_pred, y_test_pred)


def getFairnessMetric(metric_test,fairMetric):
    result = 0
    metrics = [
        metric_test.consistency(),metric_test.theil_index(),metric_test.disparate_impact(),\
        metric_test.average_abs_odds_difference(),metric_test.average_odds_difference(),\
        metric_test.equal_opportunity_difference(),metric_test.statistical_parity_difference(),\
        metric_test.error_rate_difference(), metric_test.differential_fairness_bias_amplification(),\
        metric_test.false_omission_rate_difference(), metric_test.false_discovery_rate_difference(),\
        metric_test.false_negative_rate_difference(), metric_test.false_positive_rate_difference(),
        metric_test.true_positive_rate_difference()
    ]
    metric_labels = [
        'Consistency', "Theil Index", "Disparate Impact", "Average abs odds Difference","Average odds Difference",
        "Equal Opportunity Difference","Statistical Parity Difference","Error Rate Difference","Differential Fairness Bias Amplification",
        "False Omission Rate Difference","False Discovery Rate Diffenence","False Negative Rate Difference","False Positive Rate Difference",
        "True Positive Rate Difference"
    ]

    for metric_label,metric in zip(metric_labels, metrics):
        if fairMetric == metric_label:
            result = np.round(metric,3)
            break
    if type(result) is np.ndarray:
        result = result[0]

    return result

def getThresholdMetric(loadDataAndModel,modelPath,protected_attr):
    thresh_arr = np.linspace(0.01,0.99,100)
    spd=[]
    eq_opp_diff=[]
    avg_odds_diff=[]
    dispImp=[]

    dataset_orig_train, dataset_orig_test,unprivileged_groups,privileged_groups,lr_model=getDataset(loadDataAndModel,modelPath,protected_attr)
    # lr_model, preprocessor, X_train, y_train, X_test, y_test = loadDataAndModels()
    # dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups = constructAIF36ODataset(X_train, y_train, X_test, y_test, protected_attr)
    
    for thresh in thresh_arr:
        y_train_pred, y_test_pred = getPredictions(dataset_orig_train.features,dataset_orig_test.features,lr_model['classification'], thresh)
        metric_train, metric_test = getClassMetricDataset(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups, y_train_pred, y_test_pred)
        metrics=[metric_test.statistical_parity_difference(),metric_test.equal_opportunity_difference(),metric_test.average_odds_difference(),metric_test.disparate_impact()]
        # for metric in metrics:
        spd.append(metrics[0])
        eq_opp_diff.append(metrics[1])
        avg_odds_diff.append(metrics[2])
        dispImp.append(metrics[3])
    return spd,eq_opp_diff,avg_odds_diff,dispImp

















