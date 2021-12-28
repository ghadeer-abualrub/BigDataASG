
from pyspark.sql import SparkSession
import sparknlp



spark = SparkSession \
    .builder \
    .appName("STTA") \
    .master("local[*]") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.3.4") \
    .config('spark.jars.packages', 'org.elasticsearch:elasticsearch-spark-20_2.12:7.16.2') \
    .getOrCreate()
spark = sparknlp.start()
df = spark.read.format("es").load("text/tweet")
print(df)

from sparknlp.base import DocumentAssembler

documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

from sparknlp.annotator import Tokenizer

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('tokenized')

from sparknlp.annotator import Normalizer

normalizer = Normalizer() \
    .setInputCols(['tokenized']) \
    .setOutputCol('normalized') \
    .setLowercase(True)


from sparknlp.annotator import LemmatizerModel

lemmatizer = LemmatizerModel() \
    .setInputCols(['normalized']) \
    .setOutputCol('lemmatized')

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

eng_stopwords = stopwords.words('english')

from sparknlp.annotator import StopWordsCleaner

stopwords_cleaner = StopWordsCleaner() \
    .setInputCols(['lemmatized']) \
    .setOutputCol('unigrams') \
    .setStopWords(eng_stopwords)

from sparknlp.annotator import NGramGenerator

ngrammer = NGramGenerator() \
    .setInputCols(['lemmatized']) \
    .setOutputCol('ngrams') \
    .setN(3) \
    .setEnableCumulative(True) \
    .setDelimiter('_')

from sparknlp.annotator import PerceptronModel

pos_tagger = PerceptronModel() \
    .setInputCols(['document', 'lemmatized']) \
    .setOutputCol('pos')

from sparknlp.base import Finisher

finisher = Finisher() \
    .setInputCols(['unigrams', 'ngrams', 'pos'])

from pyspark.ml import Pipeline

pipeline = Pipeline() \
    .setStages([documentAssembler,
                tokenizer,
                normalizer,
                lemmatizer,
                stopwords_cleaner,
                pos_tagger,
                ngrammer,
                finisher])

tweets_review = pipeline.fit(df).transform(df)
print(tweets_review)
spark.stop()
tweets_review.select(['finished_unigrams', 'finished_ngrams', 'finished_pos']).show(5)