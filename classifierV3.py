# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
# Scikit Learn components
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
from sklearn.cross_validation import train_test_split

df = pd.read_csv('csv-files/gtd_2011to2014.csv', encoding='Latin-1',low_memory=False)
keep = ['gname','natlty1','targsubtype1','region','weapsubtype1','nwound','nkill','property','attacktype1','guncertain1','nkillter','suicide']#,'iday','imonth','iyear']
labelHash = {}

def makeLabelHash(df):
    gname = df['gname'].unique()
    dic = {}
    for i in range(len(gname)):
        entry = gname[i]
        dic[entry] = i
    return dic
        
def oneHotEncode(df,dic):
    ser = pd.Series('gname',index = df.index)
    for i in ser.index:
        gname = df['gname'][i]
        ser.set_value(i,dic[gname])
    df['gname'] = ser
    return df

def subsetDF(df,keep):
    df = df[keep]
    return df

#==============================================================================
#                                Split
#==============================================================================
def splitDatasetTarget(df):
    dataset = df.drop('gname',axis=1)
    target = df['gname']
    return dataset,target

def splitTrainTest(dataset,target,test_size=0.2):
    trainx,testx,trainy,testy = train_test_split(dataset,target,test_size=test_size)
    return trainx,testx,trainy,testy
    
#==============================================================================
#                               Models
#==============================================================================
    
def getNameFromModel(clf):
    name = str(type(clf))
    name = name[name.rfind('.')+1:name.rfind("'")] #subset from last . to last '
    return name

def xgboost():
    clf = xgb.XGBClassifier(max_depth=8,nthread=8,silent=False,objective = 'multi:softmax')
    return clf
    
def randomForest():
    clf = RandomForestClassifier(max_depth=8,n_jobs=8,n_estimators=100)
    return clf
    
def ensemblePreds(classifiers,trainx,testx,trainy,testy):
    if type(classifiers) != list:
        print('Single classifier detected - calling fit and predict instead')
        return fitAndPredict(classifiers,trainx,testx,trainy,testy)
    preds = []
    order = []
    for clf in classifiers:
        order.append(getNameFromModel(clf))
        predictions = fitAndPredict(clf,trainx,testx,trainy,testy)
        predictions = predictions.astype(int)
        testy = testy.astype(int)
        preds.append(predictions)
        print(order[-1]+' accuracy: '+str(accuracy_score(testy,predictions)))
    preds = np.array(preds)
    preds = preds.T
    return preds

def ensembleFinalLayer(clf,preds,labels):
    trainx,testx,trainy,testy = splitTrainTest(preds,labels)
    clf.fit(trainx,trainy)
    pred = clf.predict(testx)
    print('Ensemble Accuracy: '+str(accuracy_score(testx,pred)))
    
#==============================================================================
#                                Run
#==============================================================================
    
def fitAndPredict(clf,trainx,testx,trainy,testy):
    clf.fit(trainx,trainy)
    preds = clf.predict(testx)
    preds = preds.astype(int)
    return preds
    
if __name__ in '__main__':
    labelHash = makeLabelHash(df)
    df = subsetDF(df,keep)
    df = df.dropna().reset_index(drop=True)
    df = oneHotEncode(df,labelHash)
    dataset,target = splitDatasetTarget(df)
    trainx,testx,trainy,testy = splitTrainTest(dataset,target)
    testy = testy.astype(int)
    clf = xgboost()
    preds = fitAndPredict(clf,trainx,testx,trainy,testy)
    print(accuracy_score(testy,preds))
    
    clfs = [xgboost(),randomForest()]
    preds = ensemblePreds(clfs,trainx,testx,trainy,testy)
    