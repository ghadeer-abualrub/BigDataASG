# BigDataASG

A massive amount of geo-tagged tweets is produced daily on twitter. Analyzing these tweets has the potential to enable event discovery, hidden patterns extraction and other useful insights within a specific area.
In this work, we provide an exploratory spatio-temporal analysis of a twitter stream utilizing the Twitter API to scan tweets with geo-tags, and because the data is so large, we used Kafka to keep track of the incoming tweets before saving them to an Elasticsearch index. The top hot tweets in this area are presented and rated when a user selects a bounding box on the interactive map, and other statistics are collected, aggregated, and reported via Spark NLP.
