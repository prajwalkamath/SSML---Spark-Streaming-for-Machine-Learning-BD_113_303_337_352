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
from preprocessing import preprocess
#conf=SparkConf()
#conf.setAppName("Sentiment Analysis")
#spark = SparkSession.builder.config(conf=SparkConf()).getOrCreate()
conf=SparkConf()
conf.setAppName("Sentiment Analysis")
sc = SparkContext(conf=conf)
#sqlContext = SQLContext(sc)
Senti_schema= StructType([       
    StructField('Sentiment', StringType(), False),
    StructField('Tweet', StringType(), False)
])
def create_df(rdd):
	spark = SparkContext.getOrCreate()
	sqlContext = SQLContext(spark)
	records = rdd.collect()
	lst=[]
	#records.show()
	
	#dicts = [i for j in records for i in list(json.loads(j).values())]
	c=0
	for i in records:
		for j in json.loads(i).values():
			lst.append(j)
	rowData = map(lambda x: Row(**x),lst) 
	DF = sqlContext.createDataFrame(rowData,Senti_schema)
	#dfFromData3.show()
	DF = preprocess(DF)
	DF.show()
ssc = StreamingContext(sc,5)
ssc.checkpoint("checkpoint_Sentiment")
dataStream=ssc.socketTextStream("localhost",6100)
dataStream.foreachRDD(create_df)
#print(type(dataStream))
#par = sc.parallelize(dataStream)
#df = sqlContext.read.json(par)
#df.show(5)
#record = dataStream.flatMap(json.loads)
#record.pprint()
#record = dataStream.flatMap(lambda x:x.split('\n'))
#record.pprint()
#dataStream.pprint()
#aDict = json.loads(dataStream)

ssc.start()    
ssc.awaitTermination()
ssc.stop()

