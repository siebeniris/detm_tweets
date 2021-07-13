# Temporal Evolution of the Migration-related Topics on Soical Media

## RDFS Model, Statistics, and SPARQL Queries

Go to **Project Website**: [https://siebeniris.github.io/TemporalTopics/](https://siebeniris.github.io/TemporalTopics/)

## Data Collection

### 1. Keywords
top50 most similar words using pre-trained word Word2Vec and fastText word embeddings, then manually verified:
`data/keywords.yaml`
* for crawling tweets 
* after preprocessing, the unigrams are used for calculating the centroid

### 2. 11 Destinations countries
The countries are where most refugees in Europe are hsoted. These countries are selected by ranking them according to the frequency of 
the asylum seekers obtained from [Eurostat](https://appsso.eurostat.ec.europa.eu/nui/show.do?dataset=migr_asyappctza&lang=en).

The chosen countries: _the United Kingdom, Germany, Spain, Poland, France, Sweden, Austria, Hungary, Switzerland, Netherlands, and Italy_ .



### 3. Crawled Tweets
- spanning 7 years (2013-2020)
- 384891 in total



## Temporal Topics

Top Topic Words: [link](https://github.com/siebeniris/detm_tweets/blob/master/topic_words/topic_words_10.csv)


## Knolwedge Base
See more details  [https://siebeniris.github.io/TemporalTopics/](https://siebeniris.github.io/TemporalTopics/)

Donwload: [MigrationsKB](https://github.com/siebeniris/detm_tweets/blob/master/migrationsKB_temporal_07132021_132954.tar.xz)
