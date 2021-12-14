from tweet_consumer import TweetsConsumer
import time

consumer = TweetsConsumer()
time.sleep(1)
while(True):
    consumer.consume_messages()
