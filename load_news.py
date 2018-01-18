from eventregistry import *
import pickle
from nltk import word_tokenize
import math
from itertools import chain
import string
from nltk.corpus import stopwords
from nltk import ne_chunk, pos_tag

class newsLoader:

    def __init__(self):
        self.er = EventRegistry(apiKey='aa130a68-76fe-43b8-b862-c721f0e1607a')
        self.articles = []

    '''
    query full articles from event registry by concept URI; you must know the exact concept name
    (can be found on eventregistry.org). Save articles to pickle.
    '''
    def query_articles_by_concept(self,concept_name):

        # setup a query: articles in English related to a given concept
        q = QueryArticles(lang='eng',
            conceptUri=self.er.getConceptUri(concept_name))

        # get total number of articles (max count of returned articles per query is 200)
        q.setRequestedResult(RequestArticlesInfo(page=1, count=200,
                                                        returnInfo=ReturnInfo(
                                                        articleInfo=ArticleInfoFlags(bodyLen=-1),
                                                        sourceInfo=SourceInfoFlags(title=True, location=True,
                                                                       sourceGroups=True))))
        # execute the query
        res = self.er.execQuery(q)
        count = res['articles']['totalResults']

        print("total article count: " + str(count))

        # calculate number of pages that will need to be requested to get all articles
        no_of_pages = math.ceil(count/200)
        print(no_of_pages)

        # execute multiple requests to fetch all pages
        articles = []
        for page_no in range(1,no_of_pages+1):
            q.setRequestedResult(RequestArticlesInfo(   page = page_no, count = 200,
                                                        returnInfo=ReturnInfo(
                                                        articleInfo=ArticleInfoFlags(bodyLen=-1),
                                                        sourceInfo=SourceInfoFlags(title=True, location=True, sourceGroups=True))))
            res = self.er.execQuery(q)
            articles.append(res)

        # preprocess (merge, tokenize etc.) the result
        if len(articles) > 0:
            arts = [art['articles']['results'] for art in articles]

        self.articles = list(chain.from_iterable(arts))

        #save articles before preprocessing
        with open('./pickle_files/articles_' + concept_name.replace(" ", "_"), 'wb') as fp:
               pickle.dump(self.articles, fp)

        #preprocess each article
        arts_preprocessed = self.preprocess_articles(self.articles)

        with open('./pickle_files/articles_' + concept_name.replace(" ", "_") + "_preprocessed", 'wb') as fp:
                pickle.dump(arts_preprocessed, fp)

    '''
       query full articles by concept URI from a specific media outlet; you must know the exact concept name
       (can be found on eventregistry.org) or use a Wikipedia page title (should be mapped to concepts one to one).
        Save articles to pickle.
    '''
    def query_articles_by_concept_plus_media(self, concept_name, media_outlet):

        # setup a query: articles in English related to a given concept
        q = QueryArticles(lang='eng',
                          conceptUri=self.er.getConceptUri(concept_name),sourceUri=self.er.getNewsSourceUri(media_outlet))

        # get total number of articles (max count of returned articles per query is 200)
        q.setRequestedResult(RequestArticlesInfo(page=1, count=200,
                                                 returnInfo=ReturnInfo(
                                                     articleInfo=ArticleInfoFlags(bodyLen=-1),
                                                     sourceInfo=SourceInfoFlags(title=True, location=True,
                                                                                sourceGroups=True))))
        res = self.er.execQuery(q)
        count = res['articles']['totalResults']

        print("total article count: " + str(count))

        # calculate number of pages that will need to be requested to get all articles
        no_of_pages = math.ceil(count / 200)
        print(no_of_pages)

        # execute multiple requests to fetch all pages
        articles = []
        for page_no in range(1, no_of_pages + 1):
            q.setRequestedResult(RequestArticlesInfo(page=page_no, count=200,
                                                     returnInfo=ReturnInfo(
                                                         articleInfo=ArticleInfoFlags(bodyLen=-1),
                                                         sourceInfo=SourceInfoFlags(title=True, location=True,
                                                                                    sourceGroups=True))))
            res = self.er.execQuery(q)
            articles.append(res)

        # preprocess (merge, tokenize etc.) the result
        if len(articles) > 0:
            arts = [art['articles']['results'] for art in articles]

        self.articles = list(chain.from_iterable(arts))

        # save articles before preprocessing
        with open('./pickle_files/articles_' + concept_name.replace(" ", "_") + "_" + media_outlet.replace(" ", "_"), 'wb') as fp:
            pickle.dump(self.articles, fp)

        arts_preprocessed = self.preprocess_articles(self.articles)

        with open('./pickle_files/articles_' + concept_name.replace(" ", "_") + "_" + media_outlet.replace(" ", "_") + "_preprocessed", 'wb') as fp:
            pickle.dump(arts_preprocessed, fp)

    '''
    tokenize and preprocess article bodies
    '''
    def preprocess_articles(self,articles):

        preprocessed_arts = []
        for art in articles:
            article = {}
            article['title'] = art['title']
            article['source'] = art['source']['title']

            body = word_tokenize(art['body'])
            article['body'] = self._preprocess_article_body(body)
            print(article['title'])
            print(article['source'])
            print(article['body'][0:100])
            preprocessed_arts.append(article)

        return preprocessed_arts


    '''
    remove punctuation, words with numbers, stop words, and named entities
    '''
    def _preprocess_article_body(self,body):

        # remove punctuation
        string.punctuation += ("—‘’…")
        body = [x for x in body if not self._has_punctuation(x)]

        #remove words containing numbers
        body = [x for x in body if not self._has_numbers(x)]

        #remove stop words
        stop_words = set(stopwords.words('english'))
        body = [w for w in body if not w.lower() in stop_words]

        # tag named entities in text and remove them
        body = ne_chunk(pos_tag(body), binary=True)
        body = self._remove_named_entities(body)
        # remove part of speech tags added by ne_chunk
        body = self._remove_pos_tags(body)
        print(body)

        return body


    '''
    Merge two article files and assign each articles from each file a different color.
    (Used for case study analyses.)
    '''
    @staticmethod
    def merge_two_article_files(filename_a, filename_b, output_filename):

        with open('./pickle_files/' + filename_a, 'rb') as fp:
            articles_a = pickle.load(fp)

        with open('./pickle_files/' + filename_b, 'rb') as fp:
            articles_b = pickle.load(fp)

        for art in articles_a:
            art['color'] = 'r'

        for art in articles_b:
            art['color'] = 'b'

        articles = articles_a + articles_b

        with open('./pickle_files/' + output_filename, 'wb') as fp:
            pickle.dump(articles, fp)

    #check if a string contains number
    #https://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
    @staticmethod
    def _has_numbers(word):
        return any(char.isdigit() for char in word)

    #check if a string contains punctuation
    @staticmethod
    def _has_punctuation(word):
        return any(char in string.punctuation for char in word)

    #removes names, geospatial names etc., takes text with part of speech and ne_chunk tags as argument
    @staticmethod
    def _remove_named_entities(text):
        for word in text[:]:
            if hasattr(word, 'label'):
                if word.label() == "NE":
                    text.remove(word)
        return text

    # filter out part of speech tags and convert words to lower case
    @staticmethod
    def _remove_pos_tags(text):

        words = []
        for word in text:
            words.append(word[0].lower())

        return words

    '''
    When looking for a specific event to be queried, you can lookup events related to a concept.
    Concepts should correspond to Wikipedia entries or can be looked up on eventregistry.org.
    concept example: "2016 Summer Olympics"
    '''
    def query_events_by_concept(self,concept):

        q = QueryEvents(lang='eng',
            conceptUri=self.er.getConceptUri(concept))

        q.setRequestedResult(RequestEventsInfo(page=1, count=100,
                                                 returnInfo=ReturnInfo(
                                                     articleInfo=ArticleInfoFlags(bodyLen=-1),
                                                     sourceInfo=SourceInfoFlags(title=True, location=True,
                                                                                sourceGroups=True))))

        # execute the query
        res = self.er.execQuery(q)
        res = res['events']['results']

        with open('./pickle_files/events_' + concept.replace(" ", "_"), 'wb') as fp:
               pickle.dump(res, fp)

        return res

    '''
    When querying articles by events, exact event identificators must be used (can be looked by query_events_by_concept)
    Event_uri example: eng-3220166 (Finsbury park attack).
    QueryEventArticlesIter cannot return full article bodies, only their details and URIs. The articles are queried subsequently
    by the URIs.
    '''
    def query_articles_by_event(self, event_uri, output_file):

        #setup query
        q = QueryEventArticlesIter(event_uri)

        q.setRequestedResult(RequestEventArticles(returnInfo=ReturnInfo(
                                                articleInfo=ArticleInfoFlags(bodyLen=-1))))
        #execute query
        res = q.execQuery(self.er, lang="eng")

        #filter out uris from the response
        uris = [art['uri'] for art in res]

        #execute an additional query to request all articles with given uris
        articles = self.get_articles_by_uri(uris)

        #save the response before preprocessing for analysis purposes
        with open('./pickle_files/articles_' + output_file.replace(" ", "_"), 'wb') as fp:
            pickle.dump(articles, fp)

        arts_preprocessed = self.preprocess_articles(articles)

        with open('./pickle_files/articles_' + output_file.replace(" ", "_") + "_preprocessed", 'wb') as fp:
            pickle.dump(arts_preprocessed, fp)

    '''
    query articles by their unique resource identifiers
    '''
    def get_articles_by_uri(self,uris):

        q = QueryArticle(uris)
        # get full article bodies by setting bodyLen = -1
        q.addRequestedResult(RequestArticleInfo(returnInfo=ReturnInfo(
                                                articleInfo=ArticleInfoFlags(bodyLen=-1),
                                                sourceInfo=SourceInfoFlags(title=True, location=True,
                                                                                sourceGroups=True))))
        res = self.er.execQuery(q)
        articles = []
        for art in res:
            articles.append(res[art]['info'])

        return articles