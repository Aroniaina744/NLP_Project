# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as auc
import numpy as np
import re
from scipy import sparse

class Departement(object):
	"""docstring for Brain"""

	def fit(self,text,y):
		self.vectorizer = CountVectorizer(min_df=1,binary=True)
		X = self.vectorizer.fit_transform(text)
		self.model = RandomForestClassifier()
		self.logisticRegression = LogisticRegression(class_weight='balanced')
		self.logisticRegression.fit(X,y)
		self.model.fit(X,y)
		return self
	def predict(self,text):
		X=self.vectorizer.transform(text)
		p = self.model.predict_proba(X)
		return p[:,1]

	def predict_one(self,text):
		X=self.vectorizer.transform([text])
		p = self.model.predict_proba(X)
		p1 = self.logisticRegression.predict_proba(X)
		return (p[0,1]+p1[0,1])/2
class Process(object):
	"""docstring for Brain"""

	def fit(self,text,y):
		self.vectorizer = CountVectorizer(min_df=1,binary=True)
		X = self.vectorizer.fit_transform(text)
		self.model = RandomForestClassifier()
		self.logisticRegression = LogisticRegression(class_weight='balanced')
		self.logisticRegression.fit(X,y)
		self.model.fit(X,y)
		return self
	def predict(self,text):
		X=self.vectorizer.transform(text)
		p = self.model.predict_proba(X)
		return p[:,1]

	def predict_one(self,text):
		X=self.vectorizer.transform([text])
		p = self.model.predict_proba(X)
		p1 = self.logisticRegression.predict_proba(X)
		return (p[0,1]+p1[0,1])/2
class Document(object):
	"""docstring for Brain"""

	def fit(self,text,y):
		self.vectorizer = CountVectorizer(min_df=1,binary=True)
		X = self.vectorizer.fit_transform(text)
		self.model = RandomForestClassifier()
		self.logisticRegression = LogisticRegression(class_weight='balanced')
		self.logisticRegression.fit(X,y)
		self.model.fit(X,y)
		return self
	def predict(self,text):
		X=self.vectorizer.transform(text)
		p = self.model.predict_proba(X)
		return p[:,1]

	def predict_one(self,text):
		X=self.vectorizer.transform([text])
		p = self.model.predict_proba(X)
		p1 = self.logisticRegression.predict_proba(X)
		return (p[0,1]+p1[0,1])/2
class Edition(object):
	"""docstring for Brain"""

	def fit(self,text,y):
		self.vectorizer = CountVectorizer(min_df=1,binary=True)
		X = self.vectorizer.fit_transform(text)
		self.model = RandomForestClassifier()
		self.logisticRegression = LogisticRegression(class_weight='balanced')
		self.logisticRegression.fit(X,y)
		self.model.fit(X,y)
		return self
	def predict(self,text):
		X=self.vectorizer.transform(text)
		p = self.model.predict_proba(X)
		return p[:,1]

	def predict_one(self,text):
		X=self.vectorizer.transform([text])
		p = self.model.predict_proba(X)
		p1 = self.logisticRegression.predict_proba(X)
		return (p[0,1]+p1[0,1])/2




d2 = pd.read_csv('data2.csv',sep = ";").fillna("")

d3 = d2.replace(to_replace = np.Sale, value =1)



departement = Departement()
departement.fit(d3.text,d3.department.values)

process = Process()
process.fit(d3.text,d3.process.values)

document = Document()
document.fit(d3.text,d3.document.values)

edition = Edition()
edition.fit(d3.text,d3.edition.values)

p = departement.predict(d3.text)
def predict(msg):
	print (msg)
	return {
	    "department": departement.predict_one(msg),
	    "process": process.predict_one(msg),
	    "document": document.predict_one(msg),
		"edition": edition.predict_one(msg)
    }


def test(text):
	if departement.predict_one(text)< 0.8 :
		return "no departement"
	else :
		return "with departement"

print(predict("Sale"))
print(predict("test"))
print(predict("test test"))
