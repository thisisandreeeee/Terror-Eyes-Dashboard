# -*- coding: utf-8 -*-

import pandas as pd, numpy as np
# Scikit Learn components
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
from sklearn.cross_validation import train_test_split

import classifierV3_gname as main

TARGET = 'multiple'
REMOVE = ''

df = pd.read_csv('../csv-files/gtd_2011to2014.csv', encoding='Latin-1',low_memory=False)
keep = ['natlty1','targsubtype1','region','weapsubtype1','nwound','nkill','property','attacktype1','guncertain1','nkillter','suicide','gname']#,'iday','imonth','iyear']
labelHash = {}
#labelHashReversed = {k:v for v,k in labelHash.items()}

def addOrRemove(keep,target,REMOVE=REMOVE):
    if TARGET not in keep:
        keep.append(TARGET)
    if REMOVE:
        del keep[REMOVE]
    return keep
    
def xgboost():
    clf = xgb.XGBClassifier(max_depth=8,nthread=8,silent=False,objective = 'binary:logistic')
    return clf
    
    
if __name__ in '__main__':
    keep = addOrRemove(keep,TARGET,REMOVE)
    labelHash = main.makeLabelHash(df)
    names_to_keep = main.findGroupsWithMoreThanXAttacks(df,x=5)
    #df = substituteWitUnknown(df,names_to_keep)
    df = main.removeGroups(df,names_to_keep)
    df = main.subsetDF(df,keep)
    df['natlty1'] = df['natlty1'].fillna(-99)
    df = df.dropna().reset_index(drop=True)
    df = main.oneHotEncode(df,labelHash)
    dataset,target = main.splitDatasetTarget(df,target = TARGET)
    trainx,testx,trainy,testy = main.splitTrainTest(dataset,target)
    trainx,testx,trainy,testy = main.convertToFloat(trainx,testx,trainy,testy)
    clf = xgboost()
    preds = main.fitAndPredict(clf,trainx,testx,trainy,testy)
    accuracy = accuracy_score(testy,preds) ## 88.3%
    print(accuracy)
    accuracy = main.roundAccuracy(accuracy)
    main.saveClassifier('multiple_attacksXgboost_'+accuracy+'.pkl',clf,dataset,target)
#    
#    clfs = [xgboost(),randomForest()]
#    preds = ensemblePreds(clfs,trainx,testx,trainy,testy)
#    ensembleFinalLayer(xgboost(),preds,testy)
#    