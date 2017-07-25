a=open('task1_Noun_chunk_Identification.txt','r').read()
#print a
#print("\n\n")
o=open("feature_creation.csv","w")
import nltk
import re
from nltk.corpus import sentiwordnet as swn
stop=nltk.corpus.stopwords.words('english')
s=nltk.sent_tokenize(a)
h="words, stop words, nouns, verbs, adjectives, positive, neutral, negative, target \n"
o.write(h)
#def sentiment(s):
#	w=str(s).split(' ')
#	for i in w:
#		# i
#		k=swn.senti_synsets(i)
#		try:
#		for j in k:
#			p1=j.pos() # positive score
#			n1=j.neg() # negative score
#			polarity=p1-n1 # polarity score, create a list of the score 
#			pl.append
#		except:
#			pass

for i in range(len(s)):
	c_pos=0
	c_neg=0
	c_neu=0
	k=nltk.word_tokenize(s[i])
	for a in k:
		senti=swn.senti_synsets(a)
		#print senti
		senti_list=[]
		count=0
		sentiment=0.0
		for e in senti:
			p1=e.pos_score()
			n1=e.neg_score()
			polarity=p1-n1
			senti_list.append(polarity)
			sentiment=sentiment+polarity
		#print senti_list
		#print len(senti_list)
		if len(senti_list)==0:
			senti_list=[0.0]
		#print senti_list
		#senti_list1=[]
		#for item in senti_list:
		#	if item>=0.0:
		#		senti_list1=max(senti_list)
		#	else:
		#		senti_list1=min(senti_list)
		#print senti_list1
		polarity=[]
		senti_list2=[]
		#print senti_list
		for item in senti_list:
			senti_list2=sorted(senti_list)
		#print senti_list2
		for item in senti_list2:
			if (senti_list2[0]>=0.00):
          			polarity = max(senti_list2)
			else:
				polarity = min(senti_list2)

		#if senti_list>0.0:
		#		polarity=max(senti_list)
		#	else
		#		polarity=min(senti_list)
		#print polarity
		#print len(senti_list1)
		#print senti_list_neg
		if polarity>0.00:
			c_pos=c_pos+1
		elif(polarity<0.00):
			c_neg=c_neg+1
		else:
			c_neu=c_neu+1
	#print c_pos, ' /n' , c_neu, '/n',c_neg		
	#print sentiment
		#pos=0
		#for item in senti_list:
		#	if max(item>0:
		#		pos=pos+1
		#print pos
		#neg=0
		#for item1 in senti_list:
		#	if item1<0:
		#		neg=neg+1
		#print neg
		#senti=[]
		#if pos>neg:
		#	senti=pos
		#else: 
		#	senti=neg
		#print senti
		#try:
		#	max_list=max(senti_list)
		#except:
		#	pass
		#print max_list 
		#for item in senti_list:
		#	print item
			#if len(item)<1:
			#	print senti_list.remove(item)
				#print max(senti_list)
	#senti_list1=[]
	#if sentiment>=0:
	#	senti_list1='positive'
	#else:
	#	senti_list1='negative'
	#print senti_list1
	m=nltk.pos_tag(k)
	#print m
	c=0
	for j in m:
		if re.search(r'NN(.*)',j[1]):
			c=c+1
	#print c
	l=0
	for kk in m:
		if re.search(r'VB(.*)',kk[1]):
			l=l+1
	#print l
	adj=0
	for lk in m:
		if re.search(r'JJ(.*)',lk[1]):
			adj=adj+1
	#print adj
	d=0
	for j in k:
		#print j
		p=str(j).lower()
		if p in stop:
			d=d+1
	#print d
	sentiment=[]
	if c_pos>c_neg:
		sentiment=1
	else:
		sentiment=0
	#print sentiment
	g=str(len(k))+','+ str(d) + ',' + str(c) + ',' + str(l) + ',' + str(adj)+',' + str(c_pos) + ',' + str(c_neu)+','+str(c_neg)+','+str(sentiment)

	o.write(g)
	o.write("\n")
o.close()

# Classification algorithm

import pandas as pd
import numpy as np
from sklearn import metrics
df=pd.read_csv(r"feature_creation.csv")
#print df
x=df[['words',' stop words',' nouns',' verbs',' adjectives',' positive',' neutral',' negative']]
#print x
#print x.shape
y=df[' target ']
#print y.shape
from sklearn.model_selection import train_test_split
x_train=x[0:12]
y_train=y[0:12]
x_test=x[13:20]
y_test=y[13:20]
#print x_train.shape,y_train.shape,x_test.shape,y_test.shape
from sklearn.svm import SVC
model=SVC()
model.fit(x_train,y_train)
print (model)

predicted=model.predict(x_test)
expected=y_test
print (metrics.confusion_matrix(expected,predicted))

from sklearn.metrics import classification_report
print (classification_report(expected,predicted))


