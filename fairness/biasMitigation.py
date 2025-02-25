from aif360.algorithms.preprocessing import Reweighing,DisparateImpactRemover
from aif360.algorithms.postprocessing import EqOddsPostprocessing


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from fairness.fairnessEvaluation import *

def reweighing(unprivileged_groups,privileged_groups,dataset_orig_train,dataset_orig_test,model_name,fit_params=None):
    RW = Reweighing(
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
    )
    dataset_transf_train = RW.fit_transform(dataset_orig_train)
    if model_name == 'Logistic Regression':
        model =LogisticRegression(solver='liblinear')
        fit_params = {'logisticregression__sample_weight': dataset_transf_train.instance_weights}

    elif model_name == "SVC":
        model = SVC(kernel='linear', probability=True)
        fit_params = {'svc__sample_weight': dataset_transf_train.instance_weights}

    else:
        model = DecisionTreeClassifier()
        fit_params = {'decisiontreeclassifier__sample_weight': dataset_transf_train.instance_weights}

    new_model = make_pipeline(
        StandardScaler(),
        model
    )
    new_model.fit(dataset_transf_train.features,dataset_transf_train.labels.ravel(),**fit_params)
    y_train_pred, y_test_pred= getPredictions(dataset_transf_train.features,dataset_orig_test.features,new_model)
    metric_train,metric_test=getClassMetricDataset(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups, y_train_pred, y_test_pred)

    return metric_train, metric_test

def disparateImpactRemover(unprivileged_groups,privileged_groups,dataset_orig_train,dataset_orig_test,model_name,repair_level=1.0):
    DIR = DisparateImpactRemover(repair_level=repair_level)
    dataset_transf_train = DIR.fit_transform(dataset_orig_train)
    if model_name == 'Logistic Regression':
        model =LogisticRegression(solver='liblinear')
    elif model_name == "SVC":
        model = SVC(kernel='linear', probability=True)
    else:
        model = DecisionTreeClassifier()
    new_model = make_pipeline(
        StandardScaler(),
        model
    )
    new_model.fit(dataset_transf_train.features,dataset_transf_train.labels.ravel())
    y_train_pred, y_test_pred= getPredictions(dataset_transf_train.features,dataset_orig_test.features,new_model)
    metric_train,metric_test=getClassMetricDataset(dataset_orig_train, dataset_orig_test, unprivileged_groups, privileged_groups, y_train_pred, y_test_pred)
    return metric_train, metric_test


def eqOddsPostProcessing(unprivileged_groups,privileged_groups,dataset_orig_test,data_orig_train,model_name):
    EOPP = EqOddsPostprocessing(privileged_groups = privileged_groups,
                             unprivileged_groups = unprivileged_groups,
                             seed=42)

    if model_name == 'Logistic Regression':
            model =LogisticRegression(solver='liblinear')

    elif model_name == "SVC":
        model = SVC(kernel='linear', probability=True)

    else:
        model = DecisionTreeClassifier()

    new_model = make_pipeline(
        StandardScaler(),
        model
    )
    new_model.fit(data_orig_train.features,data_orig_train.labels.ravel())

    y_pred=new_model.predict(dataset_orig_test.features)
    dataset_orig_test_pred = dataset_orig_test.copy()
    dataset_orig_test_pred.labels = y_pred.reshape(-1,1)
    EOPP = EOPP.fit(dataset_orig_test, dataset_orig_test_pred)
    data_transf_test_pred = EOPP.predict(dataset_orig_test_pred)


    metric_test = ClassificationMetric(
        dataset_orig_test,
        data_transf_test_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    return metric_test