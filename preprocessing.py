from pyspark.ml.feature import Tokenizer, RegexTokenizer,StopWordsRemover,CountVectorizer
from pyspark.sql.functions import col, udf,regexp_replace,concat_ws
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

def tokens(df):
	
	tokenizer = Tokenizer(inputCol="Tweet",outputCol="words")
	tokenized = tokenizer.transform(df)
	return tokenized.select("Sentiment","words")
	
def stop_words(df):
	remover = StopWordsRemover(inputCol="words", outputCol="Tweet")
	removed = remover.transform(df)
	return removed.select("Sentiment","Tweet")

def idf(df):
	tokenizer = Tokenizer(inputCol="Tweet", outputCol="words")
	wordsData = tokenizer.transform(df)
	count = CountVectorizer (inputCol="words", outputCol="rawFeatures",vocabSize = 600)
	model = count.fit(wordsData)
	featurizedData = model.transform(wordsData)
	idf = IDF(inputCol="rawFeatures", outputCol="features")
	idfModel = idf.fit(featurizedData)
	rescaledData = idfModel.transform(featurizedData)
	return rescaledData.select("Sentiment", "features")

def preprocess(df):
	df = df.withColumn("Tweet",regexp_replace("Tweet",r'https?://\S+|www\.\S+',""))
	df = df.withColumn("Tweet",regexp_replace("Tweet",r'@\w+',""))
	df = df.withColumn("Tweet",regexp_replace("Tweet",r'[#,\d,\?,\!,\;,\-,\*,\.,\+,\&,\_,\$,\%,\^,\(,\),\<,\>,\/,\|,\},\{,\\,\~,\',\[,\],\:,\~,\`,","]',""))
	df = tokens(df)
	df = stop_words(df)
	df = df.withColumn("Tweet",concat_ws(" ",col("Tweet")))
	df = idf(df)
	df = df.withColumn("Sentiment",col("Sentiment").cast('int'))
	#df = countvector(df)
	return df
