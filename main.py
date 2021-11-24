#import findspark
import time
from pyspark import SparkConf,SparkContext
from pyspark.streaming import StreamingContext
import sys
import requests
#findspark.init()
conf=SparkConf()
conf.setAppName("Sentiment Analysis")
sc=SparkContext(conf=conf)
sqlContext=SQLContext(sc)
ssc=StreamingContext(sc,5)
ssc.checkpoint("checkpoint_Sentiment")
dataStream=ssc.socketTextStream("localhost",6100)
#print(type(dataStream))
#df=sqlContext.read.json(sc.parallelize([dataStream]))
#df.show(5)
dataStream.pprint()
ssc.start()    
ssc.awaitTermination()
ssc.stop()
