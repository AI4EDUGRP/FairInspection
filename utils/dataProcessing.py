import pandas as pd
import numpy as np
import os
import json
from utils.HistogramBins import *

def prepareHeatmap():
    stud_sect_ques_lists=[]
    dataPath='data/section_question_tb'
    for sect in ['Section_1.csv','Section_2.csv','Section_3.csv','Section_4.csv','Section_7.csv','Section_8.csv','Section_9.csv','Section_10.csv','Section_11.csv','Section_12.csv']:
        stud_sect_ques_lists.append(pd.read_csv(os.path.join(dataPath,sect)).set_index('useraccount_id'))

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
    return data_sections, xsections, ysections

#通过make_hitogram_bins后获得每个区间的人数（将人数大于100的限制在100内）和相应的区间后展示出Line chart 
def getPerDist(dataset):
    perfValues=dataset.performance.values
    index_bins,val_ranges = make_histogram_bins(perfValues,bin_size_calculator=customizeBinSize)
    val=[np.mean(list(val_range)) for val_range in val_ranges]
    nums =[]
    for bin in index_bins:
        # if len(bin)<=100:
        #     nums.append(len(bin))
        # else:
         nums.append(len(bin))
    return val, nums

# 得到每个性别在每个小Bins中的performance
def getGenderPerf(dataset):
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

    return bins, male_performance, female_performance

def getHispPerf(dataset):
    Y_perfs=dataset[dataset.hispanic_ethnicity=='Y'].performance.values
    N_perfs=dataset[dataset.hispanic_ethnicity=='N'].performance.values

    Y_values,Y_bins=np.histogram(Y_perfs,bins=1000)
    Y_values=list(Y_values)
    Y_bins=list(Y_bins)
    Y_values.insert(0,0)

    N_values,N_bins=np.histogram(N_perfs,bins=1000)
    N_values=list(N_values)
    N_bins=list(N_bins)
    N_values.insert(0,0)

    bins= Y_bins
    nonHisp_performance=N_values
    hisp_performance=Y_values

    bins=[str(np.round(t,3)) for t in bins]
    nonHisp_performance=[float(f) for f in nonHisp_performance]
    hisp_performance=[float(f) for f in hisp_performance]

    return bins, hisp_performance, nonHisp_performance

# 得到每个性别的performance的分布 为Boxplot做准备
def getGenderPerfBoxplotData(dataset):
    perfF=[float(val) for val in dataset[dataset.gender=='F']['performance'].values]
    perfM=[float(val) for val in dataset[dataset.gender=='M']['performance'].values]
    perfGender=[perfF,perfM]
    return perfGender

# 得到每个性别的performance的分布 为Boxplot做准备
def getRacePerfBoxplotData(dataset):
    perfCaucasian=[float(val) for val in dataset[dataset.race=='Caucasian']['performance'].values]
    perfBLM=[float(val) for val in dataset[dataset.race=='Black or African American']['performance'].values]
    perfAsian=[float(val) for val in dataset[dataset.race=='Asian']['performance'].values]
    perfMoreRace=[float(val) for val in dataset[dataset.race=='Two or More Races']['performance'].values]
    perfMoreAI=[float(val) for val in dataset[dataset.race=='American Indian']['performance'].values]
    perfUnknown=[float(val) for val in dataset[dataset.race=='Unknown']['performance'].values]
    perfHPI=[float(val) for val in dataset[dataset.race=='Hawaiian or Other Pacific Islander']['performance'].values]
    
    perfRace=[perfCaucasian,perfBLM,perfAsian,perfMoreRace,perfMoreAI,perfUnknown,perfHPI]
    return perfRace

def getHispPerfBoxplotData(dataset):
    perfHisp=[float(val) for val in dataset[dataset.hispanic_ethnicity=='Y']['performance'].values]
    perfNonHisp=[float(val) for val in dataset[dataset.hispanic_ethnicity=='N']['performance'].values]
    perfHipanic=[perfHisp,perfNonHisp]
    return perfHipanic


# 为timeline chart准备数据
def sectCols(section,datasetDemog):
    '''
        获得每个section中feature列表
    '''
    currCols = [col for col in list(datasetDemog.columns) if col.endswith(section)]
    return currCols

featureDict = {}
def getDemogFeatures(section,datasetDemog,demog):
    '''
        在每个section中，统计每个feature的值,
        然后以{feature,value}的形式保存在列表中
    '''
    currCols = sectCols(section,datasetDemog)
    
    # print(currCols)
    featureDict=[{'features':feature.split('_Section')[0],'value':np.round(datasetDemog.loc[demog][feature],3)} for feature in currCols]
    return featureDict

def getTimelineData(dataset,demog):
    datasetDemog=dataset.groupby(demog).agg({"mean",'mean'})
    datasetDemog.columns=datasetDemog.columns.droplevel(1)

    sections = ['Section 1','Section 2', 'Section 3', 'Section 4', 'Section 7', 'Section 8','Section 9', 'Section 10', 'Section 11', 'Section 12']
    features = ['question_diversity', 'times_excercises', 'activiness','correct_nums','avg_time', 'parti_modules', 'active_days', 'video_pause', 'video_seek','video_playing','video_completed','num_videos_review']
    if demog=='gender':
        featureDictF={}
        featureDictM={}
        for section in sections:
            featureDictF[section] = getDemogFeatures(section, datasetDemog, 'F')
            featureDictM[section] = getDemogFeatures(section, datasetDemog, 'M')
        return featureDictF, featureDictM, sections
    elif demog =='hispanic_ethnicity':
        featureDictHisp ={}
        featureDictNonHisp ={}
        for section in sections:
            featureDictHisp[section] = getDemogFeatures(section, datasetDemog, 'Y')
            featureDictNonHisp[section] = getDemogFeatures(section, datasetDemog, 'N')
        return featureDictHisp,featureDictNonHisp,sections
    else:
        featureDictAmericanIndian={}
        featureDictAsian={}
        featureDictBAA={}
        featureDictCaucasian={}
        for section in sections:
            featureDictAmericanIndian[section] = getDemogFeatures(section, datasetDemog, 'American Indian')
            featureDictAsian[section] = getDemogFeatures(section, datasetDemog, 'Asian')
            featureDictBAA[section] = getDemogFeatures(section, datasetDemog, 'Black or African American')
            featureDictCaucasian[section] = getDemogFeatures(section,datasetDemog, 'Caucasian')
        return featureDictAmericanIndian,featureDictAsian,featureDictBAA,featureDictCaucasian,sections

# 得到sankey数据
def getSankeyData(dataPath):
    with open(os.path.join(dataPath,'linksdf.json')) as f:
        links=json.load(f)
    f.close()

    with open(os.path.join(dataPath,'nodesdf.json')) as f:
        nodes=json.load(f)
    f.close()

    with open(os.path.join(dataPath,'colorsdf.json')) as f:
        colors=json.load(f)
    f.close()
    return links, nodes, colors