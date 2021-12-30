import json
import tweepy
import time
from kafka import KafkaProducer


class TweetsProducer:

    def __init__(self, host="localhost", port="9092", topic_name="test", interval=10, value_serializer=False):
        if not value_serializer:
            self.producer = KafkaProducer(
                bootstrap_servers=[host + ":" + port])
        else:
            self.producer = KafkaProducer(bootstrap_servers=[host + ":" + port],
                                          value_serializer=lambda x: json.dumps(x).encode('utf-8'))

        self.topic = topic_name
        # interval should be an integer, the number of seconds to wait
        self.interval = interval

        self._connect_to_tweetAPI()

    def _connect_to_tweetAPI(self):

        # twitter setup ## fill here
        consumer_key = ""
        consumer_secret = ""
        access_token = ""
        access_token_secret = ""

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
            self.producer.send(self.topic, record.encode('utf-8'))
        except Exception as e:
            print(str(e))

    def _get_twitter_data(self):
        res = self.api.search_tweets(
            q="#", lang='en-us', geocode='40.7604,-73.9840,5000mi', result_type="recent")
        
        for i in res:
            #if i.coordinates != None:
            tweet = {"id": str(i.user.id_str), "timestamp": str(i.created_at), "coordinates": str(i.coordinates), "text": str(i.text), "retweet_count": str(i.retweet_count),"followers_count": str(i.user.followers_count)}
            record = json.dumps(tweet)

            print(record)
            print("-------------------------------")
            self._send(record)

    def periodic_work(self):
        self._get_twitter_data()
        print("--------- break for 10 sec ------------")
        time.sleep(self.interval)
