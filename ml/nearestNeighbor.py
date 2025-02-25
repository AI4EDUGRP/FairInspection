from sklearn.neighbors import NearestNeighbors
# from torch import AggregationType
from ml.DataProcessing import mlDataPrepare
import pickle
import os
import pandas as pd
# from DataProcessing import *

# preprocessing = pickle.load(open('data/preprocessData/preprocessing.pkl','rb'))

def getNeigborsTable(useraccount_id,neighbors,datapath): 
    # aggDataDf, _, X_test, _, _, _, _, preprocessing=mlDataPrepare()
    aggDataDf = pd.read_csv(os.path.join('data/preprocessData','aggDataDf.csv'))
    X_test = pd.read_csv(os.path.join('data/preprocessData','X_test.csv'))
    preprocessing = pickle.load(open('data/preprocessData/preprocessing.pkl','rb'))
    user_test = pd.read_csv(os.path.join('data/preprocessData','user_test.csv'))

    idx = user_test[user_test.useraccount_id==useraccount_id].index.tolist()[0]

    #将测试集中所存在的用户在原来数据集中找出来
    studDf = pd.read_csv(datapath)
    
    # 将X_test数据集重新编号
    X_test=X_test.reset_index(drop=True)
    X_test_transformed = preprocessing.fit_transform(X_test)

    neigh = NearestNeighbors(n_neighbors=6)
    neigh.fit(X_test_transformed)
    # 找出 idx在测试集中的 neighbours
    neighborsIdx = neigh.kneighbors([X_test_transformed[idx,:]],return_distance=False)[0]
    # print('neighborsIdx',neighborsIdx)

    neighborsId = user_test.values[neighborsIdx][:,0]
    # print(neighborsId)

    neighborsPerfDf=studDf[studDf.useraccountID.isin(neighborsId)][['useraccountID','y_true','y_label','y_pred','pred_label','Flag']]
    neighborsPerfDf.columns=['useraccount_id','y_true','y_label','y_pred','pred_label','Flag']
    neighborsDf = aggDataDf[['useraccount_id','gender']].merge(neighborsPerfDf,right_on='useraccount_id',left_on='useraccount_id',how='inner')
    neighborsDf = neighborsDf.loc[:neighbors-1]
    neighborsIdx = neighborsIdx[:neighbors]
    neighborsId = neighborsId[:neighbors]
    

    return neighborsDf, idx, neighborsIdx, neighborsId

















#################################################################################
## 保存数据
#################################################################################


# X_test_transformed=preprocessing.fit_transform(X_test)
# print(X_test_transformed)
# def saveData(path):
#     with open(os.path.join(path,'preprocessing.pkl'),'wb') as files:
#         pickle.dump(preprocessing,files)

#     aggDataDf.to_csv(os.path.join(path, 'aggDataDf.csv'),index=False)
#     X_train.to_csv(os.path.join(path, 'X_train.csv'),index=False)
#     X_test.to_csv(os.path.join(path,'X_test.csv'),index=False)
#     y_train.to_csv(os.path.join(path,'y_train.csv'),index=False)
#     y_test.to_csv(os.path.join(path,'y_test.csv'),index=False)
#     A_train.to_csv(os.path.join(path, 'A_train.csv'),index=False)
#     A_test.to_csv(os.path.join(path,'A_test.csv'),index=False)
# saveData('data/preprocessData')