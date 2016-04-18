import pandas as pd, numpy as np, time, re, warnings
from collections import Counter
# Scikit Learn components
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.externals import joblib
import xgboost as xgb

gtd = pd.read_csv('csv-files/gtd_2011to2014.csv', encoding='Latin-1',low_memory=False)

labelHash = {}

algo_list = [
	# ("Extra Trees", ExtraTreesClassifier(n_estimators=100))
	("Random Forest", RandomForestClassifier(max_depth=10,n_jobs=8,n_estimators=400)),
	("Extra Trees", ExtraTreesClassifier(max_depth=10,n_jobs=8,n_estimators=300)),
	# ("Logistic Regression", LogisticRegression()),
	# ("SGD Classifier",SGDClassifier()),
	("KNeighbors",KNeighborsClassifier())
	# ("Multinomial NB",MultinomialNB()),
	# ("Gaussian NB",GaussianNB())
]
    
def run():
    start = time.time()
    warnings.filterwarnings("ignore")
    features,labels = separate_column_by_type(gtd)
    features = process_nontext(features)
    features = convertDType(features)
    classifiers = train_classifier(algo_list,features,labels)
	# compare_classifiers(classifiers,features,labels,folds=5)
	# ensemble = build_ensemble(features,labels)
    ensemble(algo_list,features,labels,False)
    print("\nTotal elapsed time: %.2f secs" % (time.time()-start))
 
def convertDType(df):
    df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
    return df
    
def ensemble(clfs,features,labels,prev_save):
	if prev_save == False:
		df = pd.DataFrame()
		for clf in clfs:
			name = clf[0]
			preds = cross_val_predict(clf[1],features,labels,cv=5)
			print('5 fold cross val accuracy of '+name+': %0.2f ' % accuracy_score(labels,preds))
			df[name] = preds
		df['target']=labels
		#save file
		df.to_csv('csv-files/ensemble dataset.csv',index=False)
		print('File saved to directory')
	else:
		df = pd.read_csv('csv-files/ensemble dataset.csv')
	print('Beginnning ensemble')
	dataset = df.drop('target',axis=1)
	target = df['target']
	clf= RandomForestClassifier(n_jobs=8,max_depth=10,n_estimators=100)
	preds = cross_val_predict(clf,dataset,target,cv=5)
	score = accuracy_score(target,preds)
	print('Ensemble accuracy is '+str(score) +'%')
	return 'Completed'

def build_ensemble(features,labels):
	file_locs = ['Extra Trees','Random Forest','KNeighbors']
	train_log = pd.DataFrame()
	for loc in file_locs:
		print("Processing: " + loc)
		clf = joblib.load('classifiers/' + loc + '.pkl')
		pred = clf.predict(features)
		df = pd.DataFrame([pred])
		if train_log.empty:
			train_log = df
		else:
			train_log = pd.concat([train_log,df], axis=1, join_axes=[train_log.index])
	
	ensemble = LogisticRegression().fit(train_log,labels)
	return ensemble

# Input: original pandas dataframe read directly from the csv file
# Description: separates the dataframe into non-text and text types
# Output: dataframe of non-text variables, dataframe of text variables, and a list of labels
def separate_column_by_type(df):
	global labelHash
	features = remove_unwanted_columns(df)
	nontext_cols = [i for i in features.columns.values if i != 'gname']
	labels = [i for i in df['gname'] if i != 'Unknown']
	temp_list_of_labels=[i for i in labels]
	counts = Counter(temp_list_of_labels)
	label_id = 0
	final_labels = []
	for label in temp_list_of_labels:
		if counts[label] < 10:
			label = "Others"
		if label in labelHash: 
			final_labels.append(labelHash[label])
		else:
			labelHash[label] = label_id
			final_labels.append(labelHash[label])
			label_id += 1
	final_labels = pd.Series(final_labels).astype('category')
	return features[features['gname'] != 'Unknown'][nontext_cols], final_labels

# Input: original pandas dataframe read directly from the csv file
# Description: removes undesired columns as specified by the user and separates text-based data
# Output: non-text and text based dataframes
def remove_unwanted_columns(df):
	unwantedColumns = ['approxdate','resolution','alternative','country','latitude','longitude','specificity','location','attacktype2','attacktype3','attacktype1_txt','attacktype2_txt','attacktype3_txt','weaptype2','weaptype3','weaptype4','weapsubtype2','weapsubtype3','weapsubtype4','weapdetail','targtype2','targtype3','targsubtype2','targsubtype3','corp2','corp3','target2','target3','natlty2','natlty3','gsubname','gname2','gname3','gsubname2','gsubname3','guncertain2','guncertain3','claim2','claim3','claimmode2','claimmode3','propextent_txt','propvalue','propcomment','nhostkid','nhostkidus','nhours','ndays','divert','kidhijcountry','ransom','ransomamt','ransomamtus','ransompaid','ransompaidus','ransomnote','hostkidoutcome','nreleased','addnotes','scite1','scite2','scite3','dbsource','targtype1_txt','targtype2_txt','targtype3_txt','targsubtype1_txt','targsubtype2_txt','targsubtype3_txt','natlty1_txt','natlty2_txt','natlty3_txt','claimmode_txt','claimmode2_txt','claimmode3_txt','weaptype1_txt','weaptype2_txt','weapsubtype2_txt','weapsubtype1_txt','weaptype3_txt','weaptype4_txt','weapsubtype4_txt','weapsubtype3_txt','hostkidoutcome_txt','country_txt','region_txt','alternative_txt','eventid','related','summary','motive']
	df.drop(unwantedColumns, axis=1, inplace=True)
	return df

# processes the non-text dataframe, and ensures the columns (or missing values) are of appropriate data type (or value)
def process_nontext(df):
	#print("Processing non-textual variables")
	start = time.time()
	df = convert_dtypes(df)
	df = handle_missing_values(df)
	#print("Time taken: %.2f secs" % (time.time() - start))
	return df

# Input: non-text dataframe
# Description: converts each column to category or integer, as specified by user
# Output: non-text dataframe, with the appropriate column types
def convert_dtypes(features):
	to_category = ['extended','crit1','multiple','targtype1','targsubtype1','natlty1','guncertain1','claimed','claimmode','compclaim','weaptype1','weapsubtype1','property','propextent','ishostkid','INT_LOG','INT_IDEO','INT_MISC','INT_ANY','region','vicinity','crit2','crit3','doubtterr','success','suicide','attacktype1','provstate','city','corp1','target1']
	to_int = ['nperps','nperpcap','nkill','nkillus','nkillter','nwound','nwoundus','nwoundte']
	for var in to_category:
		features[var] = pd.to_numeric(features[var], errors='coerce').astype('category')
	for var in to_int:
		features[var] = pd.to_numeric(features[var], errors='coerce').fillna(0).astype(np.int64)
	return features

# Input: non-text dataframe
# Description: ensures the appropriate missing values are assigned to each column (instead of assigning every missing value as 0)
# Output: non-text dataframe with appropriate missing values
def handle_missing_values(features):
	to_zero = ['targtype1','targsubtype1','natlty1','guncertain1','compclaim','weapsubtype1','claimed','multiple','crit1','provstate','city','corp1','target1']
	for item in to_zero:
		try:
			features[item].fillna(0,inplace=True)
		except ValueError:
			features[item] = features[item].cat.add_categories([0])
			features[item].fillna(0,inplace=True)
	to_negNine = ['property','ishostkid','INT_LOG','INT_IDEO','INT_MISC','INT_ANY']
	for item in to_negNine:
		try:
			features[item].fillna(-9,inplace=True)
		except ValueError:
			features[item] = features[item].cat.add_categories([-9])
			features[item].fillna(-9,inplace=True)
	try:
		features['claimmode'] = features['claimmode'].fillna(10)
	except ValueError:
		features['claimmode'] = features['claimmode'].cat.add_categories([10])
		features['claimmode'].fillna(10,inplace=True)
	try:
		features['weaptype1'] = features['weaptype1'].fillna(13)
	except ValueError:
		features['weaptype1'] = features['weaptype1'].cat.add_categories([13])
		features['weaptype1'].fillna(13,inplace=True)
	try:
		features['propextent'] = features['propextent'].fillna(4)
	except ValueError:
		features['propextent'] = features['propextent'].cat.add_categories([4])
		features['propextent'].fillna(4,inplace=True)
	return features

# Input: list of user-specified algorithms, a feature vector, and a list of labels
# Description: trains a classifier for each algorithm within the given list, based on the feature vectors and respective labels
# Output: list of trained classifiers
def train_classifier(algo_list,features,labels):
	print("\n")
	classifiers = []
	for algo in algo_list:
		start = time.time()
		model = algo[1].fit(features,labels)
		name = algo[0]
		classifiers.append((name,model))
		print(str(name) + " training time: %.2f secs" % (time.time() - start))
		joblib.dump(model,'classifiers/'+ str(name) +'.pkl')
	return classifiers

# Input: list of trained classifiers
# Description: calculates the mean accuracy using kfold cross validation, and generates the weighted F measure for each classifier in the list
# Output: void
def compare_classifiers(classifiers,features,labels,folds):
	print("\n")
	for model in classifiers:
		kfold_score = cross_val_score(model[1], features, labels, cv=folds)
		predictedList = model[1].predict(features)
		f1score = f1_score(labels,predictedList,average='weighted')
		print(str(model[0]) + " accuracy: %0.3f (+/- %0.3f)" % (kfold_score.mean(), kfold_score.std() * 2))
		print(str(model[0]) + " F1 score: %0.2f" % (f1score))

# RUN THE MAIN PROGRAM
if __name__ == "__main__":
	run()