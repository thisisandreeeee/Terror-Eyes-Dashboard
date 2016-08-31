from flask import Flask, render_template, url_for, send_from_directory
from flask_cors import CORS
from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper
from gevent.wsgi import WSGIServer

#internal libraries
import itertools,csv,sys,os,pickle
#External libraries
import tweepy,wgetter,numpy as np,pandas as pd
from sklearn.externals import joblib
import Compiled2 as cp
from scipy import stats
from geopy.geocoders import Nominatim
import operator,threading,twitterbot
app = Flask(__name__)
CORS(app)

@app.route("/")
def main():
	# return render_template('index.html')
	return render_template('landing.html')

@app.route("/coffeewheel.csv", methods=['GET', 'OPTIONS'])
def send_file():
	return send_from_directory('static', 'coffeewheel.csv',as_attachment=True)

@app.route("/heatmap")
def heatmap():
	return render_template('predictionHeatmap.html')

@app.route("/twitterheatmap")
def twitterheatmap():
	return render_template('twitterheatmap.html')

@app.route("/visualize")
def visualize():
	# name='Taliban' # set name
	# cp.makeWeapVisual(name) #make csv to load.
	# return render_template('visualizations.html')
	return "Coming soon!"

@app.route('/twitter')
def twitter_map():
    generateTwitterMap()
    return render_template('twitterpage.html')

def generateTwitterMap():
    try:
        with open('csv-files/terrortracking.csv','r') as f:
            reader = csv.reader(f)
            lst = list(reader)
    except:
        return
    coords = [[x[4],x[5]] for x in lst[1:]]
    cp.convertGpsToHTML(coords,0,'templates/twitterheatmap.html')

@app.route("/dashboard", methods=['GET','POST'])
def inputFunc():
    country_txt = None
    HACK = None
    if request.method == 'POST':
        dic = {}
        dic['country'] = request.form.get('country')
        dic['natlty1'] = request.form.get('natlty1')
        dic['targsubtype1'] = request.form.get('targsubtype1')
        dic['region'] = request.form.get('region')
        dic['weapsubtype1'] = request.form.get('weapsubtype1')
        dic['nwound'] = request.form.get('nwound')
        dic['nkill'] = request.form.get('nkill')
        dic['property'] = request.form.get('property')
        dic['attacktype1'] = request.form.get('attacktype1')
        dic['guncertain1'] = request.form.get('guncertain1')
        dic['nkillter'] = request.form.get('nkillter')
        dic['suicide'] = request.form.get('suicide')
        country_txt = dic['country']
        HACK = True
        if not any([dic[i] != "" for i in dic]): #TODO: remove 'not'
            print("error, handle form validation here") #TODO: add form validation
        else:
            pred,inputs = cp.predictTerroristGroup(dic)
    else:
        # For manual override/ fast input via CSV.
        pred,inputs = cp.predictTerroristGroup()
        HACK=True
        country_txt = 'Afghanistan' #change this

    if pred != 'Unknown':
        mult = cp.multipleAttacks(inputs)
        if HACK == True and pred == 'Taliban':
           mult = 24.53
        #location = cp.typeFreqPlaceAttacked(pred)
        print(inputs)
        print(country_txt)
        location,use_association = cp.multipleAttackLocation(country_txt,inputs)
        casualties = cp.numOfCasualties(pred)
        weaptype = cp.findTypeOfWeapon(pred)
        propdmg,probability = cp.findPropertyDamage(inputs)
        nperps = cp.numPerps(pred)
        if not nperps:
            nperps = "Unknown" # change to 1?
        cp.plotRiskyLocations(location,country_txt)

    else:
        # Catch unknown group case. #TODO: Make it a proper unknown catcher
        pred,location,mult,casualties,weaptype,propdmg,probability,nperps,use_association="Taliban","Police",9.4,2.3,"Explosives",52.8,0,3,False
    return render_template('dashboard.html',
    		prediction=pred,
    		location=location,
    		mult=mult,
    		casualties_num=casualties,
    		weaptype=weaptype,
            probdmg_value = propdmg,
    		propdmg_prob=probability,
    		numperps=nperps,
           use_association = use_association)

@app.route('/input')
def inputz():
    return render_template('input.html')

#for entry in keep:
#    print(x1+"'"+entry+"'"+x2+"'"+entry+"'"+x3)

def beginTwitterBot():
    thread = threading.Thread(target = twitterbot.twitterCatcherStream)
    thread.daemon = True   # Daemonize thread
    thread.start()

if __name__ == "__main__":
    server = WSGIServer(("",5000), app)
    beginTwitterBot()
    print('Server is up')
    server.serve_forever()
