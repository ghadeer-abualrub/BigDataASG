
from pyspark.sql import SQLContext
from pyspark.ml.feature import CountVectorizer, StopWordsRemover
from kafka import KafkaConsumer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark import RDD
from pyspark import SparkContext
from pyspark.streaming import StreamingContext


from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, Tokenizer
from pyspark.ml.clustering import LDA, LDAModel
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.annotator import *
from sparknlp.common import RegexRule
from sparknlp.base import *
from nltk import *
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

import json
import random
import time
import os

spark = SparkSession \
    .builder \
    .appName("STTA") \
    .master("local[*]") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.4") \
    .config('spark.jars.packages', 'org.elasticsearch:elasticsearch-spark-20_2.12:7.16.2') \
    .getOrCreate()

spark.sparkContext.setLogLevel("warn")
spark = sparknlp.start()
# pipeline = PretrainedPipeline('explain_document_dl', lang='en')

df = spark.read.format("es").load("text/tweet")
print(df)

# Spark NLP requires the input dataframe or column to be converted to document.
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document") \
    .setCleanupMode("shrink")
# Split sentence to tokens(array)

# tokenizer = Tokenizer() \
#     .setInputCols(["document"]) \
#     .setOutputCol("token")

tokenizer = RegexTokenizer().setMinLength(5).setPattern('\\W+') \
    .setInputCols(["document"]) \
    .setOutputCol("token")
# clean unwanted characters and garbage
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")
# remove stopwords
eng_stopwords = stopwords.words('english')
stopwords_cleaner = StopWordsCleaner().setStopWords(eng_stopwords) \
    .setInputCols("normalized") \
    .setOutputCol("cleanTokens") \
    .setCaseSensitive(False)

# checker = NorvigSweetingModel.pretrained().setInputCols(['cleanTokens']).setOutputCol('checked')

# stopwords_remover = StopWordsRemover(inputCol='cleanTokens', outputCol='cleanTokens_1', stopWords=eng_stopwords)

# stem the words to bring them to the root form.
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")
# Finisher is the most important annotator. Spark NLP adds its own structure when we convert each row in the dataframe to document. Finisher helps us to bring back the expected structure viz. array of tokens.
finisher = Finisher() \
    .setInputCols(["stem"]) \
    .setOutputCols(["tokens"]) \
    .setOutputAsArray(True) \
    .setCleanAnnotations(False)
# We build a ml pipeline so that each phase can be executed in sequence. This pipeline can also be used to test the model.
nlp_pipeline = Pipeline(
    stages=[document_assembler,
            tokenizer,
            normalizer,
            stopwords_cleaner,
            stemmer,
            finisher])
# train the pipeline
nlp_model = nlp_pipeline.fit(df)
# apply the pipeline to transform dataframe.
processed_df = nlp_model.transform(df)
# nlp pipeline create intermediary columns that we dont need. So lets select the columns that we need
tokens_df = processed_df.select('timestamp' ,'retweet_count' ,'tokens')
tokens_df.show()

# ------- feature engineering -------------
cv = CountVectorizer(inputCol="tokens", outputCol="features",
                     vocabSize=500, minDF=3.0)
# train the model
cv_model = cv.fit(tokens_df)
# transform the data. Output column name will be features.
vectorized_tokens = cv_model.transform(tokens_df)
vectorized_tokens.show()
# ----------------------### Build the LDA Model------


num_topics = 4
lda = LDA(k=num_topics, maxIter=10)
model = lda.fit(vectorized_tokens)
ll = model.logLikelihood(vectorized_tokens)
lp = model.logPerplexity(vectorized_tokens)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))

# ------------Visualize the topics---------
# extract vocabulary from CountVectorizer
vocab = cv_model.vocabulary
topics = model.describeTopics()
topics.show()

topics = topics.toPandas()
print(vocab)
topics_words=[]
for index in topics['termIndices']:
    topics_words.append([vocab[idx] for idx in index])
# topics_words = topics.applymap(lambda idx_list: [vocab[idx] for idx in idx_list['termIndices']]).collect()
for idx, topic in enumerate(topics_words):
    print("topic: {}".format(idx))
    print("*"*25)
    for word in topic:
        print(word)
    print("*"*25)


tokenizer1 = RegexTokenizer().setMinLength(5).setPattern('\\W+') \
    .setInputCols(["document"]) \
    .setOutputCol("token")
new_df = tokenizer1.transform(df)
new_df['topic1'] = new_df['token'].apply(lambda x : set.intersection(set(topics_words[0]),x) )


spark.stop()