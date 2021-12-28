from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer
from sparknlp.base import *
from sparknlp.annotator import *
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("Spark NLP") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.4") \
    .config("spark.kryoserializer.buffer.max", "1000M") \
    .getOrCreate()

file_location = 'data/elonmusk_tweets.csv'
file_type = "csv"

infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
    .option("inferSchema", infer_schema) \
    .option("header", first_row_is_header) \
    .option("sep", delimiter) \
    .load(file_location)
# Verify the count
print(df.count())

# Spark NLP requires the input dataframe or column to be converted to document.
document_assembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document") \
    .setCleanupMode("shrink")
# Split sentence to tokens(array)
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")
# clean unwanted characters and garbage
normalizer = Normalizer() \
    .setInputCols(["token"]) \
    .setOutputCol("normalized")
# remove stopwords
stopwords_cleaner = StopWordsCleaner() \
    .setInputCols("normalized") \
    .setOutputCol("cleanTokens") \
    .setCaseSensitive(False)
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
tokens_df = processed_df.select('created_at', 'tokens')
tokens_df.show()


# ------- feature engineering -------------
cv = CountVectorizer(inputCol="tokens", outputCol="features",
                     vocabSize=500, minDF=3.0)
# train the model
cv_model = cv.fit(tokens_df)
# transform the data. Output column name will be features.
vectorized_tokens = cv_model.transform(tokens_df)

# ----------------------### Build the LDA Model------

num_topics = 3
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
topics_rdd = topics.rdd
topics_words = topics_rdd \
    .map(lambda row: row['termIndices']) \
    .map(lambda idx_list: [vocab[idx] for idx in idx_list]) \
    .collect()
for idx, topic in enumerate(topics_words):
    print("topic: {}".format(idx))
    print("*"*25)
    for word in topic:
        print(word)
    print("*"*25)
