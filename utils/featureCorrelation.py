import numpy as np
from utils.HistogramBins import *


# 根据demog_features统计non-sensitive features
def getNonSensFeatures(data,demog,nonSensFeature):
    demogVals = data[demog].unique()
    featureList=[]
    for demogVal in demogVals:
        featureList.append((demogVal,data[data[demog]==demogVal][nonSensFeature].values))
    return featureList

def nonSensFeatDistGender(data,nonSensFeature='question_diversity',bin_size_calculator=FreedmanDiaconisBinSize):
    featList = getNonSensFeatures(data,'gender',nonSensFeature)
    binF,_=make_histogram_bins(featList[0][1],bin_size_calculator)
    barsF=[len(bin) for bin in binF]
    axisF = list(range(len(barsF)))

    binM,_= make_histogram_bins(featList[1][1],bin_size_calculator)
    barsM = [len(bin) for bin in binM]
    axisM = list(range(len(barsM)))

    # the results are the number of students in each bin
    return barsF, barsM 