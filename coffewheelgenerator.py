# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 19:48:45 2016

@author: waffleboy
"""

#Make coffee wheel csv for locations. - FULL csv with gname and stuff.
import pandas as pd
df = pd.read_csv('csv-files/gtd_2011to2014_SITFINALS.csv',low_memory = False)

masterHash = {}
skipList = []
column = 'targtype1_txt'
filename = 'csv-files/locations.csv'
            
def parseRelated(related):
    if pd.isnull(related):
        return False
        
    if ',' in related:
        return [int(x) for x in related.split(',')]
   
    return [int(related)]
    
    
for i in df.index:
    if i in skipList:
        continue
    
    gname = df['gname'][i]
    multiple = df['multiple'][i]
    related = df['related'][i]
    targetColumn = df[column][i]
    
    if pd.isnull(targetColumn):
        continue
    
    related = parseRelated(related)
    
    targetColumn = targetColumn.replace('-','')
    if multiple and related:
        newDF = df[df['eventid'].isin(related)]
        for j in newDF.index:
            targetColumn = df[column][i]
            targetColumn = targetColumn.replace('-','')
            targetColumn2 = df[column][j]
            targetColumn2 = targetColumn2.replace('-','')
            targetColumn = targetColumn+'-'+targetColumn2
            skipList.append(j)
            
            if gname not in masterHash:
                masterHash[gname] = {targetColumn:1}
            else:
                if targetColumn not in masterHash[gname]:
                    masterHash[gname][targetColumn] = 1
                else:
                    masterHash[gname][targetColumn] += 1
        continue
            
    
    if gname not in masterHash:
        masterHash[gname] = {targetColumn:1}
    else:
        if targetColumn not in masterHash[gname]:
            masterHash[gname][targetColumn] = 1
        else:
            masterHash[gname][targetColumn] += 1


lst = []

for key,value in masterHash.items():
    for category, count in value.items():
        lst.append([key,category,count])
        
df = pd.DataFrame(lst)
df.columns = ['gname',column,'count']
df.to_csv(filename,index=False)
    
    
    
    
## TWO LEVEL WITHOUT GNAME
masterHash = {}
skipList = []
column = 'targtype1_txt'
filename = 'static/coffeewheel_master_locations.csv'
df = pd.read_csv('csv-files/gtd_2011to2014_SITFINALS.csv',low_memory = False)
  
for i in df.index:
    if i in skipList:
        continue
    
    multiple = df['multiple'][i]
    related = df['related'][i]
    targetColumn = df[column][i]
    
    if pd.isnull(targetColumn):
        continue
    
    related = parseRelated(related)
    
    targetColumn = targetColumn.replace('-','')
    if multiple and related:
        newDF = df[df['eventid'].isin(related)]
        for j in newDF.index:
            targetColumn = df[column][i]
            targetColumn = targetColumn.replace('-','')
            targetColumn2 = df[column][j]
            targetColumn2 = targetColumn2.replace('-','')
            targetColumn = targetColumn+'-'+targetColumn2
            skipList.append(j)
            
            if targetColumn not in masterHash:
                masterHash[targetColumn] = 1
            else:
                masterHash[targetColumn] += 1
        continue
    
    if targetColumn not in masterHash:
        masterHash[targetColumn] = 1
    else:
        masterHash[targetColumn] += 1

df = pd.DataFrame.from_dict(masterHash,orient='index')
df.to_csv(filename)

#
#
#def color():
#    import random
#    r = lambda: random.randint(0,255)
#    return ('#%02X%02X%02X' % (r(),r(),r()))
#
#   
#s = """"""
#
#for i in df.index:
#    s+= '"'+i+'"'+':' + '"'+color()+'"'+','