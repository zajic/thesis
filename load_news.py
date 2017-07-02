from eventregistry import *
import pickle
import bnc_lsa
from nltk import word_tokenize

my_api_key = 'aa130a68-76fe-43b8-b862-c721f0e1607a'

#query full articles from event registry (a dummy query for now) and save to pickle
def query_article():

    er = EventRegistry(apiKey=my_api_key)
    q = QueryArticles(lang='eng',
        conceptUri=er.getConceptUri("Islam"))

    # return the list of top 10 articles, including the full body (bodyLen set to 100000)
    q.setRequestedResult(RequestArticlesInfo(page=1, count=10,
                                            returnInfo=ReturnInfo(
                                            articleInfo=ArticleInfoFlags(bodyLen=100000))))
    res = er.execQuery(q)

    print(res)
    with open('./pickle_files/articles_islam', 'wb') as fp:
           pickle.dump(res, fp)

#TODO: this probably won't be used but leaving it for now
def fetch_and_save_articles():

    er = EventRegistry(apiKey = my_api_key)

    q = QueryArticlesIter(conceptUri = er.getConceptUri("Donald Trump"),lang='eng')
    articles = q.execQuery(er,sortBy = "date")

    articles_array = []
    for art in articles:
          # print(er.format(art))
          articles_array.append(art)
    print(articles_array)

    with open('./pickle_files/news', 'wb') as fp:
           pickle.dump(articles_array, fp)

#load articles from pickle file
def load_articles(filename):

    path = './pickle_files/' + filename

    with open(path, 'rb') as fp:
        news = pickle.load(fp)

    return news['articles']['results']

def list_sources_of_articles():

    news = load_articles()
    sources = []
    for art in news:
        sources.append(art['source']['title'])

    sources = set(sources)
    for source in (sorted(sources)):
        print(source)

#this is printing -1, something is wrong
def daily_available_requests():

    er = EventRegistry(apiKey=my_api_key)
    print(er.getDailyAvailableRequests())
    # print(er.getRemainingAvailableRequests())

#load article from pickle and prepare it for emotion tagging
def preprocess_article(output_filename):

    #save preprocesseda article to this path
    path = './pickle_files/' + output_filename

    #load articles saved to file
    articles = load_articles('articles_islam')
    body = articles[7]['body']

    body = word_tokenize(body)
    body = bnc_lsa.preprocess(body)

    print("preprocessed article:\n")
    print(body)

    with open(path, 'wb') as fp:
           pickle.dump(body, fp)

#query_article()
preprocess_article('chance_for_europe_art2')