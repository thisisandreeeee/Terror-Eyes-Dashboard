# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 17:57:09 2016

@author: waffleboy
"""

import pandas as pd, numpy as np
from sklearn.externals import joblib

df = pd.read_csv('../csv-files/gtd_2011to2014.csv', encoding='Latin-1',low_memory=False)
# filter out multiple and unknown
df = df[df['multiple'] == 1]
df = df[df['gname'] != 'Unknown']

## main. use assosication rule if have. if not, use frequency based.
def associationRule(df):
    dic = {}
    global b
    for i in df.index:
        country = df['country_txt'][i]
        gname = df['gname'][i]
        targtype1 = df['targtype1_txt'][i]
        relatedID = df['related'][i]
        # if singular, take that row and add. if not, loop another time then add all.
        singular = singularOrMultiple(relatedID)
        if singular:
            eventID = int(relatedID)
            secondPlaceAttacked = getSecondPlaceAttacked(df,eventID)
            if secondPlaceAttacked:
                dic = addToDic(dic,gname,targtype1,secondPlaceAttacked,country)
        else:
            relatedIDs = relatedID.split(',')
            relatedIDs = removeAllEmpty(relatedIDs)
            for entry in relatedIDs:
                eventID = int(entry)
                secondPlaceAttacked = getSecondPlaceAttacked(df,eventID)
                if secondPlaceAttacked:
                    dic = addToDic(dic,gname,targtype1,secondPlaceAttacked,country)
    return dic
    
def getSecondPlaceAttacked(df,relatedID):
    try:
        SecondPlaceAttacked = df[df['eventid'] == relatedID]
        if len(SecondPlaceAttacked) > 0:
            SecondPlaceAttacked = SecondPlaceAttacked.targtype1_txt.values[0]
        else:
            SecondPlaceAttacked = ''
    except:
        SecondPlaceAttacked = ''
    return SecondPlaceAttacked
    
def singularOrMultiple(relatedID):
    try:
        x = int(relatedID)
        return True
    except:
        return False

def addToDic(dic,gname,targtype1,secondPlaceAttacked,country):
    if country not in dic:
        dic[country] = {gname:{targtype1:{secondPlaceAttacked:1}}}
    else:
        if gname not in dic[country]:
            dic[country][gname] = {targtype1:{secondPlaceAttacked:1}}
        else:
            if targtype1 not in dic[country][gname]:
                dic[country][gname][targtype1] = {secondPlaceAttacked:1}
            else:
                if secondPlaceAttacked not in dic[country][gname][targtype1]:
                    dic[country][gname][targtype1][secondPlaceAttacked] = 1
                else:
                    dic[country][gname][targtype1][secondPlaceAttacked] += 1
    return dic
        
def removeAllEmpty(relatedIDs):
    while '' in relatedIDs:
        relatedIDs.remove('')
    return relatedIDs


def savePickleFile(dic,name):
    joblib.dump(dic,name)
    
if __name__ == '__main__':
    dic = associationRule(df)          
    savePickleFile(dic,'../dics/other_place_attacked_association.pkl')
