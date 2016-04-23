from flask import Flask, render_template, url_for, send_from_directory
from flask.ext.cors import CORS
from datetime import timedelta  
from flask import make_response, request, current_app  
from functools import update_wrapper


#internal libraries
import itertools,csv,sys,os,pickle
#External libraries
import tweepy,wgetter,numpy as np,pandas as pd
from sklearn.externals import joblib
import Compiled2 as cp
from scipy import stats
from geopy.geocoders import Nominatim
import operator
app = Flask(__name__)
CORS(app)

@app.route("/")
def main():
	return render_template('index.html')

@app.route("/dashboard")
def dashboard():

	pred = cp.predictTerroristGroup()
	if pred != 'Unknown':
		mult = float(cp.multipleAttacks(pred))*100
		location = cp.typeFreqPlaceAttacked(pred)
		casualties = cp.numOfCasualties(pred)
		weaptype = cp.findTypeOfWeapon(pred)
		propdmg = float(cp.findPropertyDamage(pred))*100
		nperps = cp.numPerps(pred)
		if not nperps:
			nperps = "Unknown"
		cp.plotRiskyLocations(location)
	else:
		location,mult,casualties,weaptype,propdmg,nperps="Police",9.4,2.3,"Explosives",52.8,3
	
	return render_template('dashboard.html',
		prediction=pred,
		location=location,
		mult=mult,
		casualties_num=casualties,
		weaptype=weaptype,
		propdmg_prob=propdmg,
		numperps=nperps)

@app.route("/coffeewheel.csv", methods=['GET', 'OPTIONS'])
def send_file():  
	return send_from_directory('static', 'coffeewheel.csv',as_attachment=True)

@app.route("/heatmap")
def heatmap():
	return render_template('predictionHeatmap.html')

@app.route("/visualize")
def visualize():
	name='Taliban' # set name
	cp.makeWeapVisual(name) #make csv to load.
	return render_template('visualizations.html')

if __name__ == "__main__":
	app.run(debug=True)
