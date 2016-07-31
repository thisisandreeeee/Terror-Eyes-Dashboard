# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, time, re, warnings
from collections import Counter
# Scikit Learn components
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('csv-files/gtd_2011to2014.csv', encoding='Latin-1',low_memory=False)
labelHash = {}
keep = ['gname','natlty1','targsubtype1','region','weapsubtype1','nwound','nkill','property','attacktype1','guncertain1','nkillter','suicide']#,'iday','imonth','iyear']
df = df[keep]

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

def splitDatasetTarget(df):
    dataset = df.drop('gname',axis=1)
    target = df['gname']
    return dataset,target

def splitTrainTest(dataset,target):
    trainx,testx,trainy,testy = train_test_split(dataset,target)
    return trainx,testx,trainy,testy

def runXgboost(trainx,testx,trainy,testy):
    clf = xgb.XGBClassifier(max_depth=6,nthread=8,silent=False,objective = 'multi:softmax')
    clf.fit(trainx,trainy)
    preds = clf.predict(testx)
    preds = preds.astype(int)
    testy = testy.astype(int)
    print(accuracy_score(testy,preds))

if __name__ in '__main__':
    labelHash = makeLabelHash(df)
    df = df.dropna().reset_index(drop=True)
    df = oneHotEncode(df,labelHash)
    dataset,target = splitDatasetTarget(df)
    trainx,testx,trainy,testy = splitTrainTest(dataset,target)
    runXgboost(trainx,testx,trainy,testy)