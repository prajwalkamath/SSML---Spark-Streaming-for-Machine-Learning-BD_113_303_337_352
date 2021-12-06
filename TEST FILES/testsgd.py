#import findspark
import time
from pyspark import SparkConf,SparkContext
from pyspark.sql.types import StructType,StructField,StringType
from pyspark.sql import SparkSession,SQLContext,Row
from pyspark.sql.functions import col
from pyspark.streaming import StreamingContext
import sys
import json
import requests
import pickle
from preprocessing import preprocess
from models import multinomial,sgd
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,confusion_matrix
import pickle
import numpy as np
accuracy = 0
sg = pickle.load(open("sgdclassifier.pickle",'rb'))
#from clustering import kminibatch
conf = SparkConf()
conf.setAppName("Sentiment Analysis")
sc = SparkContext(conf=conf)
#Creating a Global Schema
Senti_schema = StructType([       
    StructField('Sentiment', StringType(), False),
    StructField('Tweet', StringType(), False)
])
acc = open("sgd_acc.txt","w+")
f = open("sgd_f.txt","w+")
re = open("sgd_recall.txt","w+")
pr = open("sgd_precision.txt","w+")
cm = open("sgd_conmat","w+")
def create_df(rdd):
	spark = SparkContext.getOrCreate()
	sqlContext = SQLContext(spark)
	records = rdd.collect()
	lst = []
	c=0
	#Adding the records to a List
	for i in records:
		for j in json.loads(i).values():
			lst.append(j)
	#Creating Dataframe
	rowData = map(lambda x: Row(**x),lst) 
	DF = sqlContext.createDataFrame(rowData,Senti_schema)
	if DF.count() >0:
		DF = preprocess(DF)
		senti = np.array(DF.select("Sentiment").collect())
		feature = np.array(DF.select("features").collect())
		new_senti = np.squeeze(senti)
		new_feature = np.squeeze(feature)
		multimodel = sg.predict(new_feature)
		#print("Completed")
		accuracy = accuracy_score(new_senti, multimodel)
		fone = f1_score(new_senti, multimodel, average = 'weighted')
		rec = recall_score(new_senti, multimodel, average = 'weighted')
		pre = precision_score(new_senti, multimodel, average = 'weighted')
		con = confusion_matrix(new_senti, multimodel)
		acc.write(f'{accuracy}\n')
		f.write(f'{fone}\n')
		re.write(f'{rec}\n')
		pr.write(f'{pre}\n')
		cm.write(f'{con}\n')
ssc = StreamingContext(sc,5)
ssc.checkpoint("checkpoint_Sentiment")
dataStream = ssc.socketTextStream("localhost",6100)
#For each RDD calling create_df which creates a Dataframe
dataStream.foreachRDD(create_df)
ssc.start()    
ssc.awaitTermination()
ssc.stop()

