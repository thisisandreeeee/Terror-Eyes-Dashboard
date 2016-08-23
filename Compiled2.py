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
from gmaps_creds import gmaps_key
import operator
import requests
import urllib
from HTMLText import *

country = None
heatmapVariable=None


"""
Input: Features of attack, from file input.csv
Output: [String] Terrorist Group name
"""
def predictTerroristGroup(dic = {}):
    keep = ['natlty1','targsubtype1','region','weapsubtype1','nwound','nkill','property','attacktype1','guncertain1','nkillter','suicide']#,'iday','imonth','iyear']
    def format_inputs():
        df = pd.read_csv('input.csv')
        global country
        country = df['country_txt'][0]
        nontext_df,labels = separate_column_by_type(df)
        nontext_df = process_nontext(nontext_df)
        return nontext_df

    def predict_group(features):
         labelHash = joblib.load('labelHashxgb.pkl')
         classifier = joblib.load('xgboost76.pkl')
#         labelHash = joblib.load('dics/labelHashRF.pkl') #temporary hack to get this shit working
#         classifier = joblib.load('classifiers/randomforest.pkl')
         pred = classifier.predict(features)#[0]
         #pred_proba = classifier.predict_proba(features)[0]
         res = "Unknown"
         for entry in labelHash:
             if labelHash[entry] == pred:
                 res = entry
         return res
    if dic:
        global country
        try:
            country = dic['country']
            del dic['country']
        except:
            country='Afghanistan'
        dic = {k:int(v) for k,v in dic.items()}
        inputs = np.array([dic[x] for x in keep]).reshape((1,11))
        inputs = pd.DataFrame(inputs,columns=keep)
    else:
        inputs = format_inputs()
        inputs = inputs[keep]

    inputs = inputs.apply(pd.to_numeric)
    prediction = predict_group(inputs)
    inputs = addPredToInputs(inputs,prediction)
    print('Likely terrorist group: '+prediction)
    return prediction,inputs

def addPredToInputs(inputs,prediction):
    labelHash = joblib.load('labelHashxgb.pkl')
    inputs['gname'] = labelHash[prediction]
    return inputs

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

## MODIFIED
def multipleAttacks(inputs):
    classifier = joblib.load('classifiers/multiple_attacksXgboost_88.pkl')
    value = float(classifier.predict_proba(inputs)[0][0])
    value*=100
    value = round(value,2)
    if value > 50:
        print('MULTIPLE ATTACKS LIKELY with probability: '+str(value))
    else:
        print('Multiple attacks are unlikely with probability: '+str(100-value))
    return value

def multipleAttackLocation(country,inputs):
    association = True
    #input is an INT. need to convert back and forth walao.
    def convertToTargType(targsubtype1):
        converter = joblib.load('dics/targsubtype_to_targsubtype_txt.pkl')
        targsubtype1_txt = converter[targsubtype1]
        dic = joblib.load('dics/targsubtype_to_targtype.pkl')
        return dic[targsubtype1_txt]

    labelHash = joblib.load('labelHashxgb.pkl')
    labelHash_invert = {v:k for k,v in labelHash.items()}
    gname = labelHash_invert[inputs['gname'].iloc[0]]
    targtype1 = convertToTargType(inputs['targsubtype1'].iloc[0])
    dic = joblib.load('dics/other_place_attacked_association.pkl')
    try:
        places = dic[country][gname][targtype1] # should be a dictionary.
        place = max(places, key = places.get)
    except:
        association = False
        place = typeFreqPlaceAttacked(gname)
        print(place)
    return place,association

## DEPRECATED. NOT IN USE.
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

## MODIFIED
def findPropertyDamage(inputs):
    dic = {1 : 'Catastrophic (likely > $1 billion)',
            2 : 'Major (likely > $1 million but < $1 billion)',
            3 : 'Minor (likely < $1 million)',
            4 : 'Unknown'}
    classifier = joblib.load('classifiers/propertyDamageXgboost_87.pkl')
    try:
        value = classifier.predict(inputs)[0]
        probability = classifier.predict_proba(inputs)[0]
        probability = probability[probability.argmax()]
    except:
        value = 4
        probability = 'NIL'
    value = dic[value]
    print("\n")
    return value,probability

 ##Modified for dashboard --> return data instead of plotting it in gmaps.
def plotRiskyLocations(name,country_txt = ''):
    global country
    if name == None:
        print('Location is None! check your code / location and try again')
        return
    if country_txt == '':
        country_txt = country
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
    gmaps_url = "https://maps.googleapis.com/maps/api/geocode/json?key={}&address={}"
    geolocator = Nominatim()
    location=[]
    if type(name) != list:
        name = [name]
    for entry in name:
        address = urllib.parse.quote(entry + ' ' + country_txt)
        r = requests.get(gmaps_url.format(gmaps_key, address))
        resp = r.json()
        if resp['status'] == 'OK':
            for res in resp['results']:
                coords = res['geometry']['location']
                location.append([coords['lat'], coords['lng']])
        else:
            print("WTFFFFFF WAI NUUUUUU")
    convertGpsToHTML(location,0,'templates/predictionHeatmap.html')
#Writes GPS coordinates into a HTML file.
"""
 Input: List of lists of GPS coordinates.
 Output: HTML file of the heatmap.
"""
def convertGpsToHTML(data,state,csvloc):
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
        f = open(csvloc,'w')
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
