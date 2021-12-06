import time
import pickle
from pyspark import SparkConf,SparkContext
from pyspark.sql.types import StructType,StructField,StringType
from pyspark.sql import SparkSession,SQLContext,Row
from pyspark.sql.functions import col
from pyspark.streaming import StreamingContext
import sys
import json
import requests
from preprocessing import preprocess
from models import sgd
from kmeans import kmean
conf = SparkConf()
conf.setAppName("Sentiment Analysis")
sc = SparkContext(conf=conf)
#Creating a Global Schema
Senti_schema = StructType([       
    StructField('Sentiment', StringType(), False),
    StructField('Tweet', StringType(), False)
])
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
		km = kmean(DF)
		pickle.dump(km,open('kmeans.pickle','wb'))
ssc = StreamingContext(sc,5)
ssc.checkpoint("checkpoint_Sentiment")
dataStream = ssc.socketTextStream("localhost",6100)
#For each RDD calling create_df which creates a Dataframe
dataStream.foreachRDD(create_df)
ssc.start()    
ssc.awaitTermination()
ssc.stop()

