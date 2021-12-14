import json
import tweepy
import time
from kafka import KafkaProducer

class TweetsProducer():

    def __init__(self, host="localhost", port="9092", topic_name="test", interval=30, value_serializer=False):
        if not value_serializer:
            self.producer = KafkaProducer(bootstrap_servers=[host+":"+port])
        else:
            self.producer = KafkaProducer(bootstrap_servers=[host+":"+port],value_serializer=lambda x: json.dumps(x).encode('utf-8'))

        self.topic = topic_name
        #interval should be an integer, the number of seconds to wait
        self.interval = interval

        self._connect_to_tweetAPI()

    def _connect_to_tweetAPI(self):

        # twitter setup ## fill here
        consumer_key = "fDWZ7bd7l2iYzyZdeZbz2Kv0s"
        consumer_secret = "u1FsSOGTCPKPXeQXvHltK0yu1Oo2utjTYJEWRAxcEqM0Yghn3K"
        access_token = "939519071380525058-csGMphZlQY25exMW9zZ6LH3nrs1e6xJ"
        access_token_secret = "Lj52HnyIpPpFhSvnVyq0yfjKpfqTKCt7vNfmQHd3p7HdP"

        try:
            # Creating the authentication object
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            # Setting your access token and secret
            auth.set_access_token(access_token, access_token_secret)
            # Creating the API object by passing in auth information
            self.api = tweepy.API(auth)
            print("connected to tweepy API")
        except Exception as e:
            print(str(e))

    def _send(self, record):
        try:
            self.producer.send(self.topic,  record.encode('utf-8'))
        except Exception as e:
            print(str(e))

    def _get_twitter_data(self):
        res = self.api.search_tweets(q ="#", lang='en-us',result_type="recent")
        for i in res:

            tweet = {"id": str(i.user.id_str ) , "timestamp": str(i.created_at), "coordinates": str(i.coordinates),
                     "text": str(i.text), "retweet_count": str(i.retweet_count), "followers_count": str(i.user.followers_count)}
            record = json.dumps(tweet)

            print(record)
            print("-------------------------------")
            self._send(record)

    def periodic_work(self):
        self._get_twitter_data()
        time.sleep(self.interval)
