from flask import Flask, render_template, url_for

#internal libraries
import itertools,csv,sys,os,pickle
#External libraries
import gmaps,tweepy,wgetter,numpy as np,pandas as pd
from sklearn.externals import joblib
from TrainClassifier import labelHash, separate_column_by_type, process_nontext
import Compiled as cp
from scipy import stats
from geopy.geocoders import Nominatim
import operator
app = Flask(__name__)

@app.route("/")
def main():
	pred = cp.predictTerroristGroup()
	mult = cp.multipleAttacks(pred)
	location = cp.typeFreqPlaceAttacked(pred)
	casualties = cp.numOfCasualties(pred)
	weaptype = cp.findTypeOfWeapon(pred)
	propdmg = cp.findPropertyDamage(pred)
	nperps = cp.numPerps(pred)
	return render_template('dashboard.html',
    	prediction=pred,
    	location=location,
    	mult=mult,
    	casualties_num=casualties,
    	weaptype=weaptype,
    	propdmg_prob=propdmg,
    	numperps_arr=nperps)

if __name__ == "__main__":
	app.run(debug=True)