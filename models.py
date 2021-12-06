from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.linear_model import PassiveAggressiveClassifier
import numpy as np

def multinomial(df):
	clf = MultinomialNB()
	senti = np.array(df.select("Sentiment").collect())
	feature = np.array(df.select("features").collect())
	new_senti = np.squeeze(senti)
	new_feature = np.squeeze(feature)
	clf.partial_fit(new_feature,new_senti,classes = np.unique(new_senti))
	return clf

def sgd(df):
	clf = linear_model.SGDClassifier()
	senti = np.array(df.select("Sentiment").collect())
	feature = np.array(df.select("features").collect())
	new_senti = np.squeeze(senti)
	new_feature = np.squeeze(feature)
	clf.partial_fit(new_feature,new_senti,classes = np.unique(new_senti))
	return clf

def PAC(df):
	clf = PassiveAggressiveClassifier()
	senti = np.array(df.select("Sentiment").collect())
	feature = np.array(df.select("features").collect())
	new_senti = np.squeeze(senti)
	new_feature = np.squeeze(feature)
	clf.partial_fit(new_feature,new_senti,classes = np.unique(new_senti))
	return clf
