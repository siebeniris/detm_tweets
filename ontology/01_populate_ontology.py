import json
from datetime import datetime
from ast import literal_eval

import rdflib
import pandas as pd
import numpy as np
import uuid

from rdflib.namespace import Namespace, RDF
from rdflib import URIRef, Literal

### load the graph from current owl file.
g = rdflib.Graph()
# g.parse('input/migrationsKB_schema.owl', format='application/rdf+xml')
g.parse('input/migrationsKB_schema.ttl', format='ttl')

# define namespace
sioc = Namespace('http://rdfs.org/sioc/ns#')
sioc_t = Namespace('http://rdfs.org/sioc/types#')
rdfs = Namespace('http://www.w3.org/2000/01/rdf-schema#')
rdf = Namespace('https://www.w3.org/1999/02/22-rdf-syntax-ns#')
nee = Namespace('http://www.ics.forth.gr/isl/oae/core#')
schema = Namespace('http://schema.org/')
onyx = Namespace('http://www.gsi.dit.upm.es/ontologies/onyx/ns#')
wna = Namespace('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#')
dc = Namespace('http://purl.org/dc/elements/1.1/')
MGKB = "https://siebeniris.github.io/MGKB#"
mgkb = Namespace(MGKB)

## binding
g.bind('mgkb', mgkb)
g.bind("sioc", sioc)
g.bind("sioc_t", sioc_t)
g.bind('rdf', rdf)
g.bind("rdfs", rdfs)
g.bind("wna", wna)
g.bind("nee", nee)
g.bind("dc", dc)
g.bind("schema", schema)
g.bind("onyx", onyx)

### defined individuals
neutral_emotion = URIRef('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#neutral-emotion')
negative_emotion = URIRef('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#negative-emotion')
positive_emotion = URIRef('http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#positive-emotion')


def define_entity_resources(entities_dict):
    """
    rdfs:Resource (wikipedia urls)
    """
    for idx, ent_dict in entities_dict.items():
        ent_instance = URIRef(ent_dict['url'])
        ent_label = ent_dict['entity']
        ent_description = ent_dict['description']
        g.add((ent_instance, RDF.type, rdfs.Resource))  # individual of rdfs.Resource
        g.add((ent_instance, rdfs.label, Literal(ent_label)))
        g.add((ent_instance, schema.description, Literal(ent_description)))


def define_topics(topic_df):
    topics_dict = dict(tuple(topic_df.groupby('Topic')))
    # 0-9
    for topic_nr, temporal_topic in topics_dict.items():
        topic_idx = 'topic_'+str(topic_nr)
        instance =URIRef(MGKB+topic_idx)
        g.add((instance, RDF.type, mgkb.Topic))
        g.add((instance, sioc.id, Literal(str(topic_nr))))

        for time, words in tuple(zip(temporal_topic['Time'], temporal_topic['Topic Words'])):
            temp_topic_idx = f'temporal_topic_{topic_nr}_{time}'
            temp_topic_instance = URIRef(MGKB+temp_topic_idx)
            g.add((temp_topic_instance, RDF.type, mgkb.TemporalTopic))
            g.add((temp_topic_instance, schema.description, Literal(words)))
            g.add((temp_topic_instance, dc.date, Literal(time)))
            g.add((instance, mgkb.topicOccur, temp_topic_instance))



def add_triples_for_one_tweet(g, row, entities_dict):
    row_dict = row.to_dict()

    idx = str(row['id'])
    # t_idx for ontology.
    t_idx = 't' + idx
    instance = URIRef(MGKB + t_idx)  # define the identifier for the instance of Post
    g.add((instance, RDF.type, sioc.Post))  # add instance of type Post
    g.add((instance, sioc.id, Literal(idx)))  # add sioc:id
    created_at = row['created_at']  # date created
    if created_at is not None:
        g.add((instance, dc.created, Literal(created_at)))  # add dc:created
    # u_id for ontology
    u_id_gen = row['author_id_gen']  # add userAccount
    # user_id
    u_idx = f"u{u_id_gen}"
    user_instance = URIRef(MGKB + u_idx)
    # print(user_instance)
    g.add((user_instance, RDF.type, sioc.UserAccount))
    g.add((user_instance, sioc.id, Literal(u_id_gen)))  # userAccount has sioc:id.
    g.add((instance, sioc.has_creator, user_instance))  # has creator

    # place schema.
    p_id = 'p' + idx
    geo = row['geo']

    if str(geo)!='nan':
        geo_dict = literal_eval(geo)
        country_code = row['country_code']
        place_instance = URIRef(MGKB + p_id)
        g.add((place_instance, RDF.type, schema.Place))  # place individual of schema:Place
        g.add((place_instance, schema.addressCountry, Literal(country_code)))  # place individual has country code

        if 'coordinates' in geo_dict:
            lat = geo_dict['coordinates']['coordinates'][0]
            lon = geo_dict['coordinates']['coordinates'][0]
            g.add((place_instance, schema.latitude, Literal(lat)))
            g.add((place_instance, schema.longitude, Literal(lon)))

        if 'place' in geo_dict:
            place_fullname= geo_dict['place']['full_name']
            g.add((place_instance, sioc.name, Literal(place_fullname)))

        g.add((instance, schema.location, place_instance))  # has location

    #### annotated entities by Twitter, for hashtags and user mentions
    entities_twi = row['entities']

    if str(entities_twi) != 'nan':
        entities_twi = literal_eval(entities_twi)
        if 'mentions' in entities_twi:
            user_mentions = entities_twi['mentions']
            user_mentions_dict = {'m' + str(idx) + '_' + str(mid): user_mention['username'] for mid, user_mention in
                                     enumerate(user_mentions)}
            # print(user_mentions_dict)

            for mid_, username in user_mentions_dict.items():
                user_mention_instance = URIRef(MGKB + mid_)
                g.add((user_mention_instance, RDF.type, sioc.UserAccount))
                g.add((user_mention_instance, sioc.name, Literal(username)))
                g.add((instance, schema.mentions, user_mention_instance))

        if 'hashtags' in entities_twi:
            hashtags= entities_twi['hashtags']
            hashtags_dict = {'h' + str(idx) + '_' + str(hid): hashtag['tag'] for hid, hashtag in
                             enumerate(hashtags)}
            # print(hashtags_dict)
            for hid, hashtag in hashtags_dict.items():
                hashtag_instance = URIRef(MGKB + hid)
                g.add((hashtag_instance, RDF.type, sioc_t.Tag))
                g.add((hashtag_instance, rdfs.label, Literal(hashtag)))
                g.add((instance, schema.mentions, hashtag_instance))  # rdfs:label

    # Topic.
    if not np.isnan(row['topic']):
        topic_nr = row['topic']
        time = row['year']
        temp_topic_idx = f'temporal_topic_{topic_nr}_{time}'
        temp_topic_instance = URIRef(MGKB + temp_topic_idx)
        g.add((instance, dc.subject, temp_topic_instance))

    # nee:Entity
    ent_mentions = [literal_eval(v) for e, v in row_dict.items() if e.startswith('entity_') if str(v) != 'nan']
    ### entity mention dict
    if len(ent_mentions) > 0:
        ents_mention_dict = {'em' + idx + '_' + str(entid): mention for entid, mention in enumerate(ent_mentions)}
        for entid, ent in ents_mention_dict.items():
            mention_instance = URIRef(MGKB + entid)
            # print('mention instance:', mention_instance)
            mention = ent['mention']  ## detectedAs.
            ent_idx = ent['id']
            rank_score = ent['score']
            url_ = entities_dict[str(ent_idx)]['url']  # hasMatchedURI
            # print('url: ', url_)
            g.add((mention_instance, RDF.type, nee.Entity))
            g.add((mention_instance, nee.hasMatchedURI, URIRef(url_)))
            g.add((mention_instance, nee.detectedAs, Literal(mention)))
            # get confidence of the mention.
            g.add((mention_instance, nee.confidence, Literal(rank_score)))
            g.add((instance, schema.mentions, mention_instance))

    # schema: interactionStatistics
    # if not np.isnan(row['public_metrics']):
    public_metrics = literal_eval(row['public_metrics'])

    ### like count
    like_count = public_metrics['like_count']
    like_instance = URIRef(MGKB + 'like' + idx)
    g.add((like_instance, RDF.type, schema.IneractionCounter))
    g.add((like_instance, schema.interactionType, schema.LikeAction))
    g.add((like_instance, schema.userInteractionCount, Literal(like_count)))
    g.add((instance, schema.interactionStatistics, like_instance))

    share_count = public_metrics['retweet_count']
    share_instance = URIRef(MGKB + 'share' + idx)
    g.add((share_instance, RDF.type, schema.IneractionCounter))
    g.add((share_instance, schema.interactionType, schema.ShareAction))
    g.add((share_instance, schema.userInteractionCount, Literal(share_count)))
    g.add((instance, schema.interactionStatistics, share_instance))

    reply_count = public_metrics['reply_count']
    # quote_count = int(row['quote_count'])
    reply_instance = URIRef(MGKB + 'reply' + idx)
    # quote_instance = URIRef(MGKB +'quote'+idx)
    g.add((reply_instance, RDF.type, schema.IneractionCounter))
    g.add((reply_instance, schema.interactionType, schema.ReplyAction))
    g.add((reply_instance, schema.userInteractionCount, Literal(reply_count)))
    g.add((instance, schema.interactionStatistics, reply_instance))

    ### sentiment
    senti_instance = URIRef(MGKB + 'senti' + idx)
    pred_sentiment = row['pred_sentiment']
    g.add((senti_instance, RDF.type, onyx.Emotion))

    if pred_sentiment == 0:
        g.add((senti_instance, onyx.hasEmotionCategory, negative_emotion))
    if pred_sentiment == 1:
        g.add((senti_instance, onyx.hasEmotionCategory, neutral_emotion))
    if pred_sentiment == 2:
        g.add((senti_instance, onyx.hasEmotionCategory, positive_emotion))


    es_instance = URIRef(MGKB + 'es' + idx)
    g.add((es_instance, RDF.type, onyx.EmotionSet))
    g.add((es_instance, onyx.hasEmotion, senti_instance))
    g.add((instance, onyx.hasEmotionSet, es_instance))



if __name__ == '__main__':
    df = pd.read_csv('input/tweets_temporal_relevant_genid.csv', low_memory=False)
    # df = df.sample(5)

    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H%M%S")

    print('loading entities dictionary....')
    with open('../data/entities_dict_extracted_detm_20210712.json') as file:
        entities_dict = json.load(file)

    ## entity resources
    define_entity_resources(entities_dict)

    print('loading topics....')
    df_topcis = pd.read_csv('../topic_words/topic_words_10.csv')
    ### topics define in kb
    define_topics(topic_df=df_topcis)



    count = 0
    for idx, row in df.iterrows():
        print('processing ...', count, '\r')
        add_triples_for_one_tweet(g, row, entities_dict)
        count += 1

    g.serialize(f"output/migrationsKB_temporal_{date_time}.ttl", format="turtle")
