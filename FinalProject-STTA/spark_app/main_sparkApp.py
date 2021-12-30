
from pyspark.sql import SQLContext
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml.feature import CountVectorizer, HashingTF, IDF, Tokenizer
from pyspark.ml.clustering import LDA, LDAModel
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
from sparknlp.annotator import *
from sparknlp.common import RegexRule
from sparknlp.base import *
import nltk
# nltk.download('stopwords')
from nltk import *
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType, BooleanType
import pyspark.sql.functions as F
import sparknlp

from PyQt5.QtWidgets import QApplication
from main_gui import MainWindow
import pandas as pd

# nltk.download()
import sys

findspark.init('C:/Spark')

# ================================================================================


def checkWithinBox(point):
    '''bb is the bounding box, (ix,iy) are its top-left coordinates, and (ax,ay) its bottom-right coordinates. p is the point and (x,y) its coordinates.'''
    '''tweet object  (longitude first, then latitude)
        from map [[lat, long], [lat, long]] (left-bottom, top right)'''

    lat = point[1]
    lon = point[0]
    if b_box[0][0] <= lat and b_box[1][0] >= lat and b_box[0][1] <= lon and b_box[1][1] >= lon:
        # print("true")
        return True
    else:
        # print("false")
        return False

# ===============================================================================


def geoFilter(geoDF):
    checkCoUDF = udf(lambda z: checkWithinBox(z), BooleanType())
    geoDF = geoDF.withColumn(
        "within", checkCoUDF(col("coordinates")))
    # print(geoDF.show(3))
    filteredGeoDF = geoDF.filter(col("within") == True)
    return filteredGeoDF


def getTrendyTweets(df):
    varifiedUsersDF = df.filter(col("verified") == True)
    mostRetweeted = df.agg({"retweet_count": "max"}).collect()
    maxFavorite = df.agg({"favorite_count": "max"}).collect()
    maxFollowers = df.agg({"followers_count": "max"}).collect()

    '''w = Window.partitionBy('text')
    df = df.select('text', F.greatest(
        'retweet_count', 'favorite_count', 'followers_count').alias('max_value'))
    toptweets = df.withColumn('temp', F.max('max_value').over(w)).where(
        F.col('max_value') == F.col('temp')).drop('temp')'''
    mostRetweetedDF = df.filter(col("retweet_count") == mostRetweeted[0][0])
    maxFavoriteDF = df.filter(col("favorite_count") == maxFavorite[0][0])
    maxFollowersDF = df.filter(col("followers_count") == maxFollowers[0][0])
    print("most retweeted  ", mostRetweeted[0][0])
    print("most fav  ", maxFavorite[0][0])
    print("most max fol  ", maxFavorite[0][0])
    # mostRetweet(mostRetweetedDF)
    # maxFavoriteT(maxFavoriteDF)
    # maxFollowersU(maxFollowersDF)
    most_ret_txt = mostRetweetedDF.first()['text']
    print("most retweeted tweet ", most_ret_txt)
    most_ret_coor = mostRetweetedDF.first()['coordinates']
    # ----
    most_fav_txt = maxFavoriteDF.first()['text']
    print("\nmost fav tweet ", most_fav_txt)
    most_fav_coor = maxFavoriteDF.first()['coordinates']
    # -----
    most_foll_txt = maxFollowersDF.first()['text']
    print("\nthe tweet of the user with most followers", most_foll_txt)
    most_foll_coor = maxFollowersDF.first()['coordinates']

    return varifiedUsersDF, mostRetweeted[0][0], maxFavorite[0][0], maxFollowers[0][0], most_ret_txt, most_ret_coor,\
        most_fav_txt, most_fav_coor, most_foll_txt, most_foll_coor



# ===================================================================================
'''#############################################################################'''

'''############################### NLP Section #################################'''

'''#############################################################################'''


def topicModeling(nlpDF):
    document_assembler = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document") \
        .setCleanupMode("shrink")
    # Split sentence to tokens(array)
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
    nlp_model = nlp_pipeline.fit(nlpDF)
    # apply the pipeline to transform dataframe.
    processed_df = nlp_model.transform(nlpDF)
    # nlp pipeline create intermediary columns that we dont need. So lets select the columns that we need
    tokens_df = processed_df.select('date', 'retweet_count', 'tokens')
    # tokens_df.show()

    # ------- feature engineering -------------
    cv = CountVectorizer(inputCol="tokens", outputCol="features",
                         vocabSize=500, minDF=3.0)
    # train the model
    cv_model = cv.fit(tokens_df)
    # transform the data. Output column name will be features.
    vectorized_tokens = cv_model.transform(tokens_df)
    # vectorized_tokens.show()
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
    # topics.show()

    topics = topics.toPandas()
    # print(vocab)
    topics_words = []
    for index in topics['termIndices']:
        topics_words.append([vocab[idx] for idx in index])
    # topics_words = topics.applymap(lambda idx_list: [vocab[idx] for idx in idx_list['termIndices']]).collect()
    for idx, topic in enumerate(topics_words):
        print("topic: {}".format(idx))
        print("*"*25)
        for word in topic:
            print(word)
        print("*"*25)
    '''tokenizer1 = RegexTokenizer().setMinLength(5).setPattern('\\W+') \
        .setInputCols(["document"]) \
        .setOutputCol("token")
    new_df = tokenizer1.transform(df)
    new_df['topic1'] = new_df['token'].apply(
        lambda x: set.intersection(set(topics_words[0]), x))
    print(type(new_df))'''

    return topics_words


'''#######################################################################################'''

b_box = [[35.6751, -110.6982], [41.9677, -95.9766]]

spark = SparkSession \
    .builder \
    .appName("STTA") \
    .master("local[*]") \
    .config('spark.jars.packages', 'org.elasticsearch:elasticsearch-spark-20_2.12:7.16.2')\
    .getOrCreate()
# .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.4")\

#
# \
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext
sqlContext = SQLContext(sc)
'''this is temp should be exchanged with elastic df'''

'''q = """{
  "query": {
    "bool": {
      "must": {
        "match_all": {}
      },
      "filter": {
        "geo_bounding_box": {
          "pin.location": {
            "top_left": {
              "lat": 40.73,
              "lon": -74.1
            },
            "bottom_right": {
              "lat": 40.01,
              "lon": -71.12
            }
          }
        }
      }
    }
  }
}"""
df = spark.read.format("es").option("es.query", q).load("geopoint/_doc")
print(df.show(2))'''
path = "data/boulder_flood_geolocated_tweets.json"
df = spark.read.json(path)

selectedCols = ["created_at", "coordinates.coordinates",
                "text", "retweet_count", "user.followers_count", "user.verified", "place.country", "place.bounding_box", "favorite_count", "entities.hashtags"]  # ,"quote_count","reply_count"
dataDF = df.select(selectedCols)
dataDF = dataDF.withColumn("created_at2", expr(
    "substring(created_at, 5, length(created_at))"))
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
dataDF = dataDF.withColumn("date", to_date(
    col("created_at2"), "MMM dd HH:mm:ss ZZZZ yyyy"))
dataDF = dataDF.drop("created_at", "created_at2")
# filter by bounding box
geoDF = dataDF.filter("coordinates is not null")

# filter by current date
currDateDF = geoDF.filter(dataDF["date"] > lit('2013-12-1'))
# print(geoDF.select("hashtags.*").show(20,False))
# geoDF.printSchema()

fullFilteredDF = geoFilter(currDateDF)
varifiedUsersDF, mostRetweeted, maxFavorite, maxFollowers, most_ret_txt, most_ret_coor,\
    most_fav_txt, most_fav_coor, most_foll_txt, most_foll_coor = getTrendyTweets(
        fullFilteredDF)
topicslist = topicModeling(fullFilteredDF)
topic1DF = fullFilteredDF.where(fullFilteredDF['text'].rlike(
    "|".join(["(" + pat + ")" for pat in topicslist[0]])))
varcount11 = topic1DF.filter(col("verified") == True).count()
topic2DF = fullFilteredDF.where(fullFilteredDF['text'].rlike(
    "|".join(["(" + pat + ")" for pat in topicslist[1]])))
varcount12 = topic2DF.filter(col("verified") == True).count()
topic3DF = fullFilteredDF.where(fullFilteredDF['text'].rlike(
    "|".join(["(" + pat + ")" for pat in topicslist[2]])))
topicsCount = [topic1DF.count(), topic2DF.count(), topic3DF.count()]
varcount13 = topic3DF.filter(col("verified") == True).count()

varcount1 = varcount11+varcount12+varcount13
# ==================================================
# 4 months time span
fullFilteredDF = geoFilter(geoDF)
varifiedUsersDFF, mostRetweetedF, maxFavoriteF, maxFollowersF, most_ret_txtF, most_ret_coorF,\
    most_fav_txtF, most_fav_coorF, most_foll_txtF, most_foll_coorF = getTrendyTweets(
        fullFilteredDF)
topicslistFull = topicModeling(fullFilteredDF)
topic1DFFull = fullFilteredDF.where(fullFilteredDF['text'].rlike(
    "|".join(["(" + pat + ")" for pat in topicslist[0]])))
varcount21 = topic1DFFull.filter(col("verified") == True).count()
topic2DFFull = fullFilteredDF.where(fullFilteredDF['text'].rlike(
    "|".join(["(" + pat + ")" for pat in topicslist[1]])))
varcount22 = topic2DFFull.filter(col("verified") == True).count()
topic3DFFull = fullFilteredDF.where(fullFilteredDF['text'].rlike(
    "|".join(["(" + pat + ")" for pat in topicslist[2]])))
varcount23 = topic3DFFull.filter(col("verified") == True).count()

varcount2 = varcount21+varcount22+varcount23

topicsCountFull = [topic1DFFull.count(), topic2DFFull.count(),
                   topic3DFFull.count()]

# ==============================================================================

print('most_ret_txt ', type(most_ret_txt),
      ' most_ret_txt ', type(most_ret_txtF))


def printfromchild(txt):
    print("child txt ", txt)


print("===================================================================================================")
print("===================================================================================================")
print(most_ret_coorF[1], most_fav_coorF[1], most_foll_coorF[1],
      most_ret_coor[1], most_fav_coor[1], most_foll_coor[1], "=======", most_ret_coorF[0], most_fav_coorF[0], most_foll_coorF[0], most_ret_coor[0], most_fav_coor[0], most_foll_coor[0])
print("===================================================================================================")
print("===================================================================================================")
markers = pd.DataFrame({
    'lon': [most_ret_coorF[0], most_fav_coorF[0], most_foll_coorF[0], most_ret_coor[0], most_fav_coor[0], most_foll_coor[0]],
    'lat': [most_ret_coorF[1], most_fav_coorF[1], most_foll_coorF[1], most_ret_coor[1], most_fav_coor[1], most_foll_coor[1]],
    'name': ['Retweeted '+str(mostRetweetedF)+' times', 'Favoured '+str(maxFavoriteF)+' times', 'Has '+str(maxFollowersF) + ' followers', 'Retweeted '+str(mostRetweeted)+' times', 'Favoured '+str(maxFavorite)+' times', 'Has '+str(maxFollowers) + ' followers']
    # , 'value': [10, 12, 40, 70, 23, 43, 100, 43]
}, dtype=str)
# ===============================================================================
app = QApplication(sys.argv)
win = MainWindow(markers, topicslist, varcount1, most_ret_txt,
                 topicslistFull, varcount2, most_ret_txtF, topicsCount, topicsCountFull, most_fav_txt, most_fav_txtF)
win.show()
sys.exit(app.exec_())

# self.spark.stop()'''
