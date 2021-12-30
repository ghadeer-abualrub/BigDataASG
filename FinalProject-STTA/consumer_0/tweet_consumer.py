from kafka import KafkaConsumer
from elastic_search import TweetsElasticSearch
import time
import json


class TweetsConsumer():

    def __init__(self, host="localhost", port="9092", topic_name="test"):
        self.consumer = KafkaConsumer(topic_name, bootstrap_servers=[
                                      host+":"+port], auto_offset_reset='latest', enable_auto_commit=True, value_deserializer=lambda m: json.loads(m.decode('utf-8')))
        self.elastic_search_pointer = TweetsElasticSearch()
        self.elastic_search_pointer.create_index()
        self.messages = []

    def consume_messages(self):
        for message in self.consumer:
            message = message.value
            #print('message: ', message)
            #print(type(message))
            # add to elastic
            self.elastic_search_pointer.push_to_index(message)
            time.sleep(20)
            self.messages.append(message)
            time.sleep(20)
            #self.elastic_search_pointer.search()
