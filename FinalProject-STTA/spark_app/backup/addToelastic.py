
import json
from time import sleep
from datetime import datetime
from elasticsearch import Elasticsearch, helpers

client = Elasticsearch("localhost:9200")

"""
JSON DATA IN FILE:
{"str field": "some string", "int field": 12345, "bool field": True}
{"str field": "another string", "int field": 42, "bool field": False}
{"str field": "random string", "int field": 3856452, "bool field": True}
{"str field": "string value", "int field": 11111, "bool field": False}
{"str field": "last string", "int field": 54321, "bool field": True}
"""


def get_data(self):
    return [l.strip() for l in open(str(self), encoding="utf-8", errors='ignore')]


docs = get_data("data/boulder_flood_geolocated_tweets.json")

print("String docs length:", len(docs))


doc_list = []

for num, doc in enumerate(docs):
    try:
        doc = doc.replace("True", "true")
        doc = doc.replace("False", "false")
        dict_doc = json.loads(doc)

        dict_doc["timestamp"] = datetime.now()
        dict_doc["_id"] = num
        doc_list += [dict_doc]

    except json.decoder.JSONDecodeError as err:
        print("ERROR for num:", num, "-- JSONDecodeError:", err, "for doc:", doc)
        print("Dict docs length:", len(doc_list))

try:
    print("\nAttempting to index the list of docs using helpers.bulk()")
    resp = helpers.bulk(
        client,
        doc_list,
        index="geo2",
        doc_type="tweets"
    )

    print("helpers.bulk() RESPONSE:", resp)
    print("helpers.bulk() RESPONSE:", json.dumps(resp, indent=4))

except Exception as err:

    print("Elasticsearch helpers.bulk() ERROR:", err)
    quit()

query_all = {
    'size': 10_000,
    'query': {
        'match_all': {}
    }
}

print("\nSleeping for a few seconds to wait for indexing request to finish.")
sleep(2)

# pass the query_all dict to search() method
resp = client.search(
    index="geo2",
    body=query_all
)

#print ("search() response:", json.dumps(resp, indent=4))

# print the number of docs in index
print("Length of docs returned by search():", len(resp['hits']['hits']))

"""
Length of docs returned by search(): 5
"""
