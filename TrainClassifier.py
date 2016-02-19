import pandas as pd, numpy as np, time, re, warnings, scipy.sparse as ssp
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Scikit Learn components
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.externals import joblib

gtd = pd.read_csv('csv-files/gtd_2011to2014.csv', encoding='Latin-1',low_memory=False)

labelHash,Count_Vectorizer,Tfidf_Transformer,cachedStopwords,porter,selector = {},None,None,None,None,None

algo_list = [
	# ("Extra Trees", ExtraTreesClassifier(n_estimators=100))
    ("Random Forest", RandomForestClassifier(n_estimators=100)),
    ("Extra Trees", ExtraTreesClassifier(n_estimators=100)),
    ("Logistic Regression", LogisticRegression()),
    ("SGD Classifier",SGDClassifier()),
    ("KNeighbors",KNeighborsClassifier()),
    ("Multinomial NB",MultinomialNB()),
    ("Gaussian NB",GaussianNB())
]

def run():
	start = time.time()
	initialize()
	nontext_df,text_df,labels = separate_column_by_type(gtd)
	nontext_df = process_nontext(nontext_df)
	text_df = process_text(text_df,labels)
	# stacks the 2 dataframes horizontally, which are of different data structures
	features = ssp.hstack([nontext_df.to_sparse(),text_df]).todense()
	classifiers = train_classifier(algo_list,features,labels)
	compare_classifiers(classifiers,features,labels,folds=5)
	print("\nTotal elapsed time: %.2f secs" % (time.time()-start))

# instantiate variables that will be required later
def initialize():
	print("Initializing global variables")
	global Count_Vectorizer, cachedStopwords, porter, selector
	Count_Vectorizer = CountVectorizer(ngram_range=(1,3)) # change settings for unigram, bigram, trigram
	cachedStopwords = stopwords.words("english")
	porter = PorterStemmer()
	selector = SelectKBest(f_classif, k=1000)
	warnings.filterwarnings("ignore")

# Input: original pandas dataframe read directly from the csv file
# Description: separates the dataframe into non-text and text types
# Output: dataframe of non-text variables, dataframe of text variables, and a list of labels
def separate_column_by_type(df):
	global labelHash
	nontext_df,text_df = remove_unwanted_columns(df)
	nontext_cols = [i for i in nontext_df.columns.values if i != 'gname']
	text_cols = [i for i in text_df.columns.values if i != 'gname']
	temp_list_of_labels=[i for i in gtd.gname if i != 'Unknown']
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
	return nontext_df[nontext_df['gname'] != 'Unknown'][nontext_cols], text_df[text_df['gname'] != 'Unknown'][text_cols], final_labels

# Input: original pandas dataframe read directly from the csv file
# Description: removes undesired columns as specified by the user and separates text-based data
# Output: non-text and text based dataframes
def remove_unwanted_columns(df):
	text_df = df.copy()
	unwantedColumns = ['approxdate','resolution','alternative','country','latitude','longitude','specificity','location','attacktype2','attacktype3','attacktype1_txt','attacktype2_txt','attacktype3_txt','weaptype2','weaptype3','weaptype4','weapsubtype2','weapsubtype3','weapsubtype4','weapdetail','targtype2','targtype3','targsubtype2','targsubtype3','corp2','corp3','target2','target3','natlty2','natlty3','gsubname','gname2','gname3','gsubname2','gsubname3','guncertain2','guncertain3','claim2','claim3','claimmode2','claimmode3','propextent_txt','propvalue','propcomment','nhostkid','nhostkidus','nhours','ndays','divert','kidhijcountry','ransom','ransomamt','ransomamtus','ransompaid','ransompaidus','ransomnote','hostkidoutcome','nreleased','addnotes','scite1','scite2','scite3','dbsource','targtype1_txt','targtype2_txt','targtype3_txt','targsubtype1_txt','targsubtype2_txt','targsubtype3_txt','natlty1_txt','natlty2_txt','natlty3_txt','claimmode_txt','claimmode2_txt','claimmode3_txt','weaptype1_txt','weaptype2_txt','weapsubtype2_txt','weapsubtype1_txt','weaptype3_txt','weaptype4_txt','weapsubtype4_txt','weapsubtype3_txt','hostkidoutcome_txt','Unnamed: 134','Unnamed: 135','Unnamed: 136','country_txt','region_txt','alternative_txt','eventid','related']
	textColumns = ['gname','summary','motive']

	for col_list in [unwantedColumns,textColumns]:
		for col in col_list:
			if col != 'gname':
				try:
					df.pop(col)
				except:
					continue
	return df,text_df[textColumns]

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

# Input: text dataframe and list of labels
# Description: processes the text dataframe by cleaning (removing regex), stemming the words and converting to feature vector with tfidf counts
# Output: document of features with tfidf counts
def process_text(df,labels):
	print("Processing text-based variables")
	start = time.time()
	clean_df = clean_text_data(df)
	stemmed_df = stem_words(porter,clean_df)
	data_counts = Count_Vectorizer.fit_transform(stemmed_df)
	temp_selector = selector.fit(data_counts,labels)
	data_counts = temp_selector.transform(data_counts)
	temp_tfidf_transformer = TfidfTransformer(use_idf=True).fit(data_counts)
	Tfidf_Transformer = temp_tfidf_transformer
	tfidf_doc = TfidfTransformer(use_idf=True).fit_transform(data_counts)
	print("Time taken: %.2f secs" % (time.time() - start))
	return tfidf_doc

# applies user-specified stemming algorithm on each word
def stem_words(stemmer,df):
	if stemmer != None:
		stemmedTokens =[]
		for item in df:
			tokens = item.split(' ')
			tokens = [stemmer.stem(token) for token in tokens if not token.isdigit()]
			stemmedTokens.append(tokens)
		stemmed_df = []
		for token in stemmedTokens:
			stemmed_df.append(" ".join(str(i) for i in token))
	return stemmed_df

# removes regex and returns a combined list of words
def clean_text_data(df):
	clean_df = []
	for col in df.columns.values:
		column = []
		for item in df[col]:
			if type(item) == str:
				item = re.sub(r'[^\w\s]','',item)
				# remove stopwords and change to lowercase
				item = ' '.join([word.lower() for word in item.split() if word not in cachedStopwords])
			else:
				item=""
			column.append(item)
		clean_df.append(column)
	return [x + y for x,y in zip(clean_df[0],clean_df[1])]

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
