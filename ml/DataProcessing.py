
import pandas as pd
import numpy as np
import os
import copy

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt


from sklearn.manifold import TSNE
import pickle

# 将F设置为Privileged Group, 将M设置为unPrivileged Group
def genderEncoder(x):
    if x=='F':
        return 1
    else:
        return 0
#将Non hispanic设置为Privileged Group
def hispEncoder(x):
    if x=='N':
        return 1
    else:
        return 0
# 将Caucassian设置为Privileged Group
def raceEncoder(x):
    if x=='Caucasian':
        return 1
    else:
        return 0

# 将数据读入，为通过失败设定阈值，将数据split，并且转换 返回numerical 和categorical transformer
def mlDataPrepare(threshold=0.6,attr='gender'):
    aggDataDf = pd.read_csv('data/studAggregatedDataUpdated.csv') # 读出数据
    aggDataDf.gender=aggDataDf.gender.map(genderEncoder)
    aggDataDf.hispanic_ethnicity=aggDataDf.hispanic_ethnicity.map(hispEncoder)
    aggDataDf.race=aggDataDf.race.map(raceEncoder)
    aggDataDf['labels']=aggDataDf['performance'].apply(lambda x: 1 if x>threshold else 0 ) #根据阈值设置通过和失败Label
    # aggDataDf中包括Labels

    users =aggDataDf[['useraccount_id']] # 得到Protected variables
    X = aggDataDf.drop(labels=['performance','labels','useraccount_id'],axis=1) # 删除performance, labels和useraccount_id
    y = aggDataDf['labels'] #获得label

    if attr is None:
        X_train, X_test, y_train, y_test, user_train, user_test = train_test_split(
            X,
            y,
            users,
            test_size=0.3,
            random_state=12345
            )        
    else:
        X_train, X_test, y_train, y_test, user_train, user_test = train_test_split(
            X,
            y,
            users,
            test_size=0.3,
            random_state=12345,
            stratify=aggDataDf[attr]
            )

    numerical_transformer = Pipeline(
        steps=[
            ('scaler',StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('ohe',OneHotEncoder())
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ('num',numerical_transformer,selector(dtype_exclude='object'))
        ]
    )
    return aggDataDf, X_train, X_test, y_train, y_test, user_train, user_test, preprocessing


def savePreprocessData(preprocessPath,aggDataDf,X_train,X_test,y_train,y_test,user_train,user_test,preprocessing):
    with open(os.path.join(preprocessPath,'preprocessing.pkl'),'wb') as files:
        pickle.dump(preprocessing,files)
    
    aggDataDf.to_csv(os.path.join(preprocessPath,'aggDataDf.csv'),index=False)
    X_train.to_csv(os.path.join(preprocessPath,'X_train.csv'),index=False)
    X_test.to_csv(os.path.join(preprocessPath,'X_test.csv'),index=False)
    y_train.to_csv(os.path.join(preprocessPath,'y_train.csv'),index=False)
    y_test.to_csv(os.path.join(preprocessPath,'y_test.csv'),index=False)
    user_train.to_csv(os.path.join(preprocessPath,'user_train.csv'),index=False)
    user_test.to_csv(os.path.join(preprocessPath,'user_test.csv'),index=False)

aggDataDf, X_train, X_test, y_train, y_test, A_train, A_test, preprocessing=mlDataPrepare()

# savePreprocessData("./data/preprocessData",aggDataDf,X_train,X_test,y_train,y_test,A_train,A_test,preprocessing)


def SNEProcessor(X_test,preprocessing):
    tsne = TSNE(n_components=1,verbose=1,random_state=123)
    X_test_transform = preprocessing.fit_transform(X_test)
    x = tsne.fit_transform(X_test_transform)
    return x

model= LogisticRegression(solver='liblinear',fit_intercept=True)
def predict(model, preprocessing, X_train, y_train,X_test):
    lr_predictor = Pipeline(
        steps=[
            ('preprocessing',copy.deepcopy(preprocessing)),
            ('classification',model)
        ]
    )
    lr_predictor.fit(X_train,y_train)
    y_pred = lr_predictor.predict(X_test)
    y_pred_proba = np.round(lr_predictor.predict_proba(X_test)[:,1],3)   
    return lr_predictor,y_pred_proba,y_pred


aggDataDf, X_train, X_test, y_train, y_test, A_train, A_test, preprocessing = mlDataPrepare()
x = SNEProcessor(X_test,preprocessing)
model = LogisticRegression(solver='liblinear',fit_intercept=True)
lr_predictor,y_pred_proba,y_pred = predict(model,preprocessing, X_train, y_train, X_test)

fpr, tpr, _ = metrics.roc_curve(y_test,y_pred_proba)


# #获得X_test的实际Label
y_true = np.round(aggDataDf.loc[X_test.index,'performance'].values,3)
pred_label = (y_pred_proba>0.6).astype(int)
y_pred_score = model.score(X_test,y_test)

# 构建学生展示表
def getStudDf(origDf,X_test, X, y_true, y_label, y_pred, y_pred_score, pred_label):
    '''
        params:
            origDf表示已经通过阈值将学生的是否是通过课程进行标注后的表
            X_test是split后的测试集，该集合上没有useraccound_id，如果使用X_train同理
            X是表示通过TSNE转换后所生成的一维特征，注意：由TSNE所得到的X是二维数组需要通过X[:,0]得到一维数组
    '''
    #首先获得X中每个值所对应的useraccount_id
    X_id=[str(val) for val in origDf.loc[X_test.index,'useraccount_id'].values]
    y_true=np.round(origDf.loc[X_test.index,'performance'].values,3).tolist()
    y_label=origDf.loc[X_test.index,'labels'].values.tolist()

    #由 x_test的useraccount_id,一维特征，实际的performance, 实际的Label,预测的Performance和预测的label组成的dataFrame
    studDf=pd.DataFrame(
        {'useraccountID':X_id,
        'PrincipalFeature':X,
        'y_true':y_true,
        'y_label':y_label,
        'y_pred':y_pred,
        'y_pred_score':y_pred_score,
        'pred_label':pred_label
        }
    )
    studDf.loc[studDf[studDf.pred_label>studDf.y_label].index,'Flag']='FPS'
    studDf.loc[studDf[studDf.pred_label<studDf.y_label].index,'Flag']='FNS'

    studDfFPS=studDf[studDf.Flag=='FPS']
    studDfFNS=studDf[studDf.Flag=='FNS']
    studDfPredPass=studDf[studDf.pred_label==1]
    studDfPredFail=studDf[studDf.pred_label==0]
    studDfTruePass=studDf[studDf.y_label==1]
    studDfTrueFail=studDf[studDf.y_label==0]
    return studDf,studDfTruePass,studDfTrueFail,studDfPredPass,studDfPredFail,studDfFNS,studDfFPS

studDf, studDfTruePass, studDfTrueFail, studDfPredPass, studDfPredFail, studDfFNS, studDfFPS = getStudDf(aggDataDf, X_test, x[:,0], y_true, y_test, y_pred_proba, y_pred_score, pred_label)

def saveData(path, lr_predictor, studDf, studDfTruePass, studDfTrueFail, studDfPredPass, studDfPredFail, studDfFNS, studDfFPS):
    with open(os.path.join(path,'lr_predictor.pkl'),'wb') as files:
        pickle.dump(lr_predictor,files)

    studDf.to_csv(os.path.join(path, 'studDf.csv'),index=False)
    studDfTruePass.to_csv(os.path.join(path, 'studDfTruePass.csv'),index=False)
    studDfTrueFail.to_csv(os.path.join(path,'studDfTrueFail.csv'),index=False)
    studDfPredPass.to_csv(os.path.join(path,'studDfPredPass.csv'),index=False)
    studDfPredFail.to_csv(os.path.join(path,'studDfPredFail.csv'),index=False)
    studDfFNS.to_csv(os.path.join(path, 'studDfFNS.csv'),index=False)
    studDfFPS.to_csv(os.path.join(path,'studDfFPS.csv'),index=False)
    print('data saved...')

# saveData(path ='data/lrModel')