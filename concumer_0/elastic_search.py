from elasticsearch import Elasticsearch


class TweetsElasticSearch():

    def __init__(self, host='localhost', port=9200):
        self.host = host
        self.port = port
        # indices
        self.indices = ['time', 'geo', 'text']
        self._connect()


    def _connect(self):
        self.es = Elasticsearch([self.host], port=self.port)

        if self.es.ping():
            print("ES connected successfully")
        else:
            print("Not connected")

    def create_index(self):
        created = False
        # index settings
        mapping = [
        {
            "mappings": {
                "request-info": {
                    "properties": {
                        "timestamp": {"type": "date"}
                    }} }}
        ,{
            "mappings": {
                "request-info": {
                    "properties": {
                        "coordinates": {"type": "geo_point"}
                    }}}} ,
                {
                    "mappings": {
                        "request-info": {
                            "properties": {
                                "text": {"type": "text"}}}}}

        ]

        for i in range(len(mapping)):
            try:
                if not self.es.indices.exists(self.indices[i]):
                    # Ignore 400 means to ignore "Index Already Exist" error.
                    self.es.indices.create(index=self.indices[i], ignore=400, body=mapping[i])
                    print(self.indices[i],'index has been created successfully')
                created = True
            except Exception as ex:
                print(str(ex))

    def push_to_index(self, message):
        for index in self.indices:
            try:
                response = self.es.index(
                    index=index,
                    doc_type="tweet",
                    body=message
                )
                print("Write response is :: {}\n\n".format(response))
            except Exception as e:
                print("Exception is :: {}".format(str(e)))

    def get(self):
        resp = self.es.get(index="time", id=1)
        print(resp['_source'])

    def search(self):
        res = self.es.search(index="time", query={"match_all": {}})
        print("Got %d Hits:" % res['hits']['total']['value'])

