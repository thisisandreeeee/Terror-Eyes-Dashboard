# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
# Scikit Learn components
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import xgboost as xgb
from sklearn.cross_validation import train_test_split

df = pd.read_csv('csv-files/gtd_2011to2014_ADDED.csv', encoding='Latin-1',low_memory=False)
keep = ['gname','natlty1','targsubtype1','region','weapsubtype1','nwound','nkill','property','attacktype1','guncertain1','nkillter','suicide']#,'iday','imonth','iyear']
labelHash = {}
#labelHashReversed = {k:v for v,k in labelHash.items()}

def findGroupsWithMoreThanXAttacks(df,x=5):
    value = df['gname'].value_counts() > x
    names_to_keep = [value.index[i] for i in range(len(value)) if value[i] == True]
    return names_to_keep
    
def removeGroups(df,names_to_keep):
    return df[df['gname'].isin(names_to_keep)].reset_index(drop=True)
    
def substituteWitUnknown(df,names_to_keep):
    df['gname'][~df['gname'].isin(names_to_keep)] = 'Unknown'
    return df

def subNAwith99(df):
    df = df.fillna(-99)
    return df
    
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
    trainx,testx,trainy,testy = convertToFloat(trainx,testx,trainy,testy)
    clf.fit(trainx,trainy)
    pred = clf.predict(testx)
    print('Ensemble Accuracy: '+str(accuracy_score(testy,pred)))
    
#==============================================================================
#                                Run
#==============================================================================
    
## temp
def loadNewInputsAndPredict(clf,path):
    global labelHash
    inputs = pd.read_csv(path,encoding='Latin-1')
    inputs = inputs[keep]
    testx = inputs.drop('gname',axis=1)
    testy = inputs['gname']
    testy_translated = [labelHash[x] for x in testy]
    preds = clf.predict(testx)
    print('Testing on new data')
    print(accuracy_score(testy_translated,preds))
    print('correct index:')
    for i in range(len(preds)):
        if preds[i] == testy_translated[i]:
            print(i)
    return preds
    
    
def convertToFloat(trainx,testx,trainy,testy):
    trainx = trainx.astype(float)
    trainy = trainy.astype(float)
    testx = testx.astype(float)
    testy = testy.astype(float)
    return trainx,testx,trainy,testy
    
def fitAndPredict(clf,trainx,testx,trainy,testy):
    clf.fit(trainx,trainy)
    preds = clf.predict(testx)
    preds = preds.astype(int)
    return preds
    
if __name__ in '__main__':
    labelHash = makeLabelHash(df)
    names_to_keep = findGroupsWithMoreThanXAttacks(df,x=5)
    #df = substituteWitUnknown(df,names_to_keep)
    df = removeGroups(df,names_to_keep)
    df = subsetDF(df,keep)
    df['natlty1'] = df['natlty1'].fillna(-99)
    df = df.dropna().reset_index(drop=True)
    df = oneHotEncode(df,labelHash)
    dataset,target = splitDatasetTarget(df)
    trainx,testx,trainy,testy = splitTrainTest(dataset,target)
    trainx,testx,trainy,testy = convertToFloat(trainx,testx,trainy,testy)
    clf = xgboost()
    preds = fitAndPredict(clf,trainx,testx,trainy,testy)
    print(accuracy_score(testy,preds))
    
    clfs = [xgboost(),randomForest()]
    preds = ensemblePreds(clfs,trainx,testx,trainy,testy)
    ensembleFinalLayer(xgboost(),preds,testy)
    