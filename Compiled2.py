# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:25:43 2016

@author: Thiru
"""
#internal libraries
import itertools,csv,sys,os,pickle
#External libraries
import tweepy,wgetter,numpy as np,pandas as pd,googlemaps
from sklearn.externals import joblib
from Classifier_v2 import separate_column_by_type, process_nontext
from scipy import stats
from geopy.geocoders import Nominatim
import operator
from HTMLText import *

country = None
heatmapVariable=None


"""
Input: Features of attack, from file input.csv
Output: [String] Terrorist Group name
"""
def predictTerroristGroup():
    def format_inputs():
        df = pd.read_csv('input.csv')
        global country
        country = df['country_txt'][0]
        nontext_df,labels = separate_column_by_type(df)
        nontext_df = process_nontext(nontext_df)
        return nontext_df

    def predict_group(features):
         labelHash = joblib.load('dics/labelHash.pkl') #temporary hack to get this shit working
         classifier = joblib.load('classifiers/randomforest.pkl')
         pred = classifier.predict(features)[0]
         #pred_proba = classifier.predict_proba(features)[0]
         res = "Unknown"
         for entry in labelHash:
             if labelHash[entry] == pred:
                 res = entry
         return res

    inputs = format_inputs()
    prediction = predict_group(inputs)
    print('Likely terrorist group: '+prediction)
    return prediction

def makeWeapVisual(name):
    df = pd.read_csv('csv-files/weapons.csv',encoding='Latin-1')
    df = df[df['gname'] == name]
    newdf = df.drop('gname',axis=1)
    newdf.to_csv('static/currentWeapon.csv',index=False,header=False)

"""
Input: [String] Terrorist Group name
Output: [String] Risky Locations

Description: This function prints the details of the terrorist group, and of its next attack (if any)
"""
def printTerroristDetails(name):
    print("\n")
    print('***** Summary Details for '+name+' *****')
    print("\n")
    multipleAttacks(name)
    location=typeFreqPlaceAttacked(name)
    numOfCasualties(name)
    findTypeOfWeapon(name)
    findPropertyDamage(name)
    numPerps(name)
    return location #return risky location!

def multipleAttacks(name):
    locProb = "csv-files/prob_mult.csv"
    df2 = pd.read_csv(locProb,encoding='Latin-1')
    value = round(float(df2[df2['gname']==name]['prob_mult']),3)
    if value > 0.5:
        print('MULTIPLE ATTACKS LIKELY with probability: '+str(value))
    else:
        print('Multiple attacks are unlikely with probability: '+str(1-value))
    return value

def typeFreqPlaceAttacked(name):
    dic= pickle.load(open('dics/typeOfPlace','rb'))
    if name in dic.keys():
        likelyPlaceAttacked= max(dic[name].items(), key=operator.itemgetter(1))[0]
        confidence = dic[name][likelyPlaceAttacked] / sum(dic[name].values())
        confidence = round(confidence,3)
        print("\n")
        print('Most Frequent Place attacked: '+likelyPlaceAttacked+' consisting of '+str(confidence*100)+'% of all attacks')
    return likelyPlaceAttacked

def numOfCasualties(name):
    print('\n')
    dic = pickle.load(open('dics/numOfCasualties','rb'))
    if name in dic.keys():
        print("Estimated number of casualties: " + str(round(dic[name],3)))
    else:
        print('Number of Casualties Unknown')
    return round(dic[name],2)

def conditionalPlaceAttacked(name):
    ##TO BE DONE WHEN I CAN BE BOTHERED TO
    return

def findTypeOfWeapon(name):
    dic= pickle.load(open('dics/typeOfWeapon','rb'))
    if name in dic.keys():
        likelyAttackWeapon= max(dic[name].items(), key=operator.itemgetter(1))[0]
        confidence = dic[name][likelyAttackWeapon] / sum(dic[name].values())
        confidence = round(confidence,3)
        print("\n")
        print('Likely type of weapon used in successive attacks is '+likelyAttackWeapon+ 'with probability: '+str(confidence))
    else:
        print("\n")
        print('Likely type of weapon unknown')
    return likelyAttackWeapon


def numPerps(name):
    dic = pickle.load(open('dics/numPerps','rb'))
    if name in dic.keys():
        print("\n")
        mean= np.mean(dic[name])
        median=np.median(dic[name])
        mode= stats.mode(dic[name])
        print ('Mean Num Perpetrators = '+str(mean)+'\n Median Num Perpetrators = '+str(median) +'\n Mode Num Perpetrators = '+str(int(mode.mode[0])))
        return [mean,median,mode]
    print("\n")
    print('Likely size of attackers unknown.')
    return False

def findPropertyDamage(name):
    dic = pickle.load(open('dics/propertyDamage','rb'))
    try:
        currGroup=dic[name]
    except:
        print(name + 'will likely NOT have property damage with probability 1')
        return
    probability=currGroup[0] / sum(currGroup)
    print("\n")
    if (probability) > 0.5:
        print(name +' will likely have property damage of estimated < $1 Millon with probability '+str(round(probability,3)))
    else:
        print(name + 'will likely NOT have property damage with probability '+str(round(probability,3)))
    return round(probability,2)

 ##Modified for dashboard --> return data instead of plotting it in gmaps.
def plotRiskyLocations(name):
    global country
    if name == None:
        return
	#Consider saving this to pickle file.
    dic={'Business':['Business','Gas','mall','restaurant','cafe','hotel'],
		 'Government (General)':['Government buildings','Ministry'],
		  'Police':['Police post','prison','Police'],
		  'Military':['military base','air base','navy'],
		  'abortion related':'abortion clinic',
		  'airports & aircraft':'airport',
		  'Government (Diplomatic)': 'embassy',
		   'Educational Institution':['school','university'],
		   'Food or Water Supply':['water treatment plant','farms'],
			'NGO':['NGO','Non governmental organisations'],
			'Maritime':['port','ferry'],
			'Journalists & Media':'newspaper company',
			'Other':['fire station','hospital'],#wtf do you code for this
			'Private Citizens & Property':['shopping malls','markets'],
		   'Religious Figures/Institutions':['temples','churches','mosques'],
			'Terrorists/Non-State Militia':'militia',
			'Transportation':['Train station','Bus stations'],
			'Utilities':['power plant','water plant'],
			'Tourists':['tourist spots'],
			'Telecommunications':['Radio station','TV station','Internet provider'],
			'Violent Political Party': 'political party' #and this
	}
    name = dic[name]
    geolocator = Nominatim() #swap for google.
    location=[]
    if type(name) != list:
        loc=geolocator.geocode(name + ' '+country,exactly_one=False,timeout=10)
        if loc is not None:
            location.append(loc)
    else:
        for entry in name:
            loc = geolocator.geocode(entry + ' '+country,exactly_one=False,timeout=10)
            if loc is not None:
                location.append(loc)
    location= list(itertools.chain(*location)) #flatten list
    data=[]
    for entry in location:
        data.append([entry.latitude,entry.longitude])
    convertGpsToHTML(data,0)
#Writes GPS coordinates into a HTML file.
"""
 Input: List of lists of GPS coordinates.
 Output: HTML file of the heatmap.
"""
def convertGpsToHTML(data,state):
    #taken from HTMLText
    global initial1,initial2,twitter1,endTwitter,heatmapVariable
    #new google.maps.LatLng(-6.9742784,109.122315),
    FORMAT = 'new google.maps.LatLng('
    addToHTML=''
    for i in range(len(data)):
        lat = data[i][0]
        long = data[i][1]
        add = FORMAT + str(lat) + ','+str(long)+'),'
        addToHTML+=add
    addToHTML=addToHTML[:-1] #remove last comma
    if state==0:
        heatmapVariable = initial1+addToHTML+twitter1 #set var for future rewrites to add twitter
        f = open('templates/predictionHeatmap.html','w')
        f.write(initial1+addToHTML+initial2) #INITIAL HEATMAP WITH NO TWITTER POINTS
        f.close()
    elif state == 1:
        f = open('templates/predictionHeatmap.html','w')
        f.write(heatmapVariable+addToHTML+endTwitter) #INITIAL HEATMAP WITH NO TWITTER POINTS
        f.close()
    else:
        print('That is not a valid state')

def run():
    print('***Welcome to the Integrated Terrorism Response Solution done by Andre, Eddy, and Thiru***')
    country = input("What is your Country?: ")
    print("\n")
    predictedGroup = predictTerroristGroup()
    location = printTerroristDetails(predictedGroup)
    plotRiskyLocations(location,country)

if __name__ == '__main__':
    run()
