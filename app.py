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

# @app.route("/dashboard")
# def dashboard():
# 	pred = cp.predictTerroristGroup()
# 	if pred != 'Unknown':
# 		mult = float(cp.multipleAttacks(pred))*100
# 		location = cp.typeFreqPlaceAttacked(pred)
# 		casualties = cp.numOfCasualties(pred)
# 		weaptype = cp.findTypeOfWeapon(pred)
# 		propdmg = float(cp.findPropertyDamage(pred))*100
# 		nperps = cp.numPerps(pred)
# 		if not nperps:
# 			nperps = "Unknown"
# 		cp.plotRiskyLocations(location)
# 	else:
# 		location,mult,casualties,weaptype,propdmg,nperps="Police",9.4,2.3,"Explosives",52.8,3
#
# 	return render_template('dashboard.html',
# 		prediction=pred,
# 		location=location,
# 		mult=mult,
# 		casualties_num=casualties,
# 		weaptype=weaptype,
# 		propdmg_prob=propdmg,
# 		numperps=nperps)


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
    coords = [[x[3],x[4]] for x in lst[1:]]
    cp.convertGpsToHTML(coords,0,'templates/twitterheatmap.html')

@app.route("/dashboard", methods=['GET','POST'])
def inputFunc():
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
        if not any([dic[i] != "" for i in dic]): #TODO: remove 'not'
            print("error, handle form validation here") #TODO: add form validation
        else:
            pred,inputs = cp.predictTerroristGroup(dic)
            if pred != 'Unknown':
                mult = cp.multipleAttacks(inputs)
                #location = cp.typeFreqPlaceAttacked(pred)
                location = cp.multipleAttackLocation(dic['country'],inputs)
                casualties = cp.numOfCasualties(pred)
                weaptype = cp.findTypeOfWeapon(pred)
                propdmg,probability = cp.findPropertyDamage(pred)
                nperps = cp.numPerps(pred)
                if not nperps:
                    nperps = "Unknown"
                cp.plotRiskyLocations(location)
    else:
        pred,location,mult,casualties,weaptype,probability,propdmg,nperps="Taliban","Police",9.4,2.3,"Explosives",52.8,"Unknown",3
        return render_template('dashboard.html',
        		prediction=pred,
        		location=location,
        		mult=mult,
        		casualties_num=casualties,
        		weaptype=weaptype,
                probdmg_value = propdmg,
        		propdmg_prob=probability,
        		numperps=nperps)
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
