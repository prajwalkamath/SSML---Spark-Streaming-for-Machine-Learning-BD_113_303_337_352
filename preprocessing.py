from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType


def preprocess(df):
	tokenizer = Tokenizer(inputCol="Tweet",outputCol="words")
	countTokens = udf(lambda words: len(words), IntegerType())
	tokenized = tokenizer.transform(df)
	return tokenized.select("Sentiment","words")
