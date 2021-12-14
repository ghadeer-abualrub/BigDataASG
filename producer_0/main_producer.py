from tweet_producer import TweetsProducer
import coordinates_collector

coordinates_collector.collect()
print(coordinates_collector.coordinates_obj.coordinates)

producer_0 = TweetsProducer()

while(True):
    producer_0.periodic_work()
