#####################
# Module loads files from British National Corpus and pre-processes them (tokenize, remove stop words, special characters).
# Creates TFIDF matrix and generates U, sigma and VT (Singular Value Decomposition).
# TFIDF, terms, sigma  and VT are stored as Python "pickle" files
######################

from nltk.corpus.reader.bnc import BNCCorpusReader
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.extmath import randomized_svd
import pandas
import numpy
import scipy
import pickle
from nltk import ne_chunk, pos_tag
import time
from nltk.corpus import wordnet as wn

path_to_bnc = './corpora/bnc_larger_sample'

def main():
    #measure the time taken to process BNC (performance purposes)
    start = start = time.time()

    docs = readBNC()
    tfidf, terms = performTFIDF(docs)
    #convert tfidf matrix from condensed to dense format and transpose
    tfidf = scipy.sparse.csr_matrix.todense(tfidf)
    tfidf = numpy.transpose(tfidf)

    U, sigma, VT = singularValueDecomp(tfidf,100)

    saveMatricesToPickle(tfidf,terms,sigma,VT)
    end = time.time()
    print(end - start)

#load and parse files from British National Corpus
def readBNC():
    # Instantiate the reader
    print("Reading BNC sample...")

    #load all files in BNC corpus
    bnc_reader = BNCCorpusReader(root=path_to_bnc, fileids=r'[A-K]/\w*/\w*\.xml')

    # number of sentences in the corpus
    print(str(len(bnc_reader.sents())) + " sentences")
    # number of words in the corpus
    print(str(len(bnc_reader.words())) + " words")
    files = bnc_reader.fileids();
    #add all files to one array
    text_corpus = []
    text_unprocessed = []
    for file in files:
        text = bnc_reader.words(fileids=file,stem=False)
        text_unprocessed.append(text)
        #remove special chars, stopwords and append to an array of documents
        print("\n file" + file)

        preprocessed_text = preprocess(text)
        text_corpus.append(preprocessed_text)

        #print document sample
        print(preprocessed_text[:50])
        print("...etc.")

    print("\nnumber of documents in the corpus: " + str(len(text_corpus)))

    # print file names in the corpus
    print("documents: " + str(bnc_reader.fileids()))

    #these files are used for analytical purposes (number of words, most frequent words etc.)
    with open('./pickle_files/bnc_preprocessed', 'wb') as fp:
        pickle.dump(text_corpus, fp)

    with open('./pickle_files/bnc_unprocessed', 'wb') as fp:
        pickle.dump(text_unprocessed, fp)

    return text_corpus

#remove special characters, stop words, numbers, words shorter than 3 chars, named entities (cities, persons etc.)
# rermove words that do not exist in Wordnet dictionary, lemmatize nouns (plural to singular) - lemmatizing commented out for now
# todo: split words with a slash before processing
def preprocess(text):

    stop_words = defineStopWords()
    print("words before preprocessing: " + str(len(text)))

    # remove words shorter than 3 chars
    text = [w for w in text if len(w) > 2]
    #remove stop words
    text = [w for w in text if not w.lower() in stop_words]
    #remove numbers
    text = [w for w in text if not hasNumbers(w)]
    #remove words containing symbols such as "½"
    text = [w for w in text if not containsUnicodeFractions(w)]

    #tag named entities in text and remove them
    text = ne_chunk(pos_tag(text), binary=True)
    text = remove_named_entities(text)

    #remove part of speech tags added by ne_chunk
    text = remove_pos_tags(text)

    #remove words that do not exist in Wordnet
    final_text = [w for w in text if len(wn.synsets(w))>0]

    #lemmatize nouns
    # lemmatizer = WordNetLemmatizer()
    # final_text = [lemmatizer.lemmatize(w) for w in text]
    print("words after preprocessing: " + str(len(final_text)))

    return final_text

#https://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
def hasNumbers(word):
    return any(char.isdigit() for char in word)

#check whether a string contains unicode fractions
def containsUnicodeFractions(word):

    fractions = ["½","⅓","⅔","¼","¾","⅕","⅖","⅗","⅘","⅙","⅚","⅐","⅛","⅜","⅝","⅞","⅑","⅒"]
    return any(char in fractions for char in word)

#amend  the list of stopwords with more words to be excluded
def defineStopWords():

    #TODO: extend this list with words that have definitely no affective meaning or remove this snippet whatsoever
    stop_words = set(stopwords.words('english'))
    months = ["january","february","march","april","may","june","july","august","september","october","november","december"]
    days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    other = ["yesterday","today","tomorrow"]
    stop_words |= set(days)
    stop_words |= set(months)

    return stop_words

#returns TFIDF matrix and an array of unique terms in a set of documents, and prints TFIDF matrix as pandas dataframe (with word labels)
def performTFIDF(docs):

    # convert words from an array to plaintext for each doc (todo: is this really necessary? couldn't the vectorizer just read an array of words)
    for i in range(len(docs)):
        docs[i] = " ".join(docs[i])

    # initialize the vectorizer
    transformer = TfidfVectorizer()
    tfidf = transformer.fit_transform(docs)

    #convert to panda dataframe
    #TODO: this maybe will be deleted in the future
    tfidf_panda = pandas.DataFrame(tfidf.toarray().transpose(), index=transformer.get_feature_names())

    print("\nTFIDF as pandas dataframe:")
    print(tfidf_panda)
    terms = transformer.get_feature_names()
    return tfidf, terms


def singularValueDecomp(tfidf_matrix,components):

    U, sigma, VT = randomized_svd(tfidf_matrix,
                                  n_components=components,
                                  n_iter=20,
                                  random_state=123)

    print("\n singular value decomposition: \n")
    print("\n" + str(U))
    print("\n" + str(sigma))
    print("\n" + str(VT))

    return U,sigma,VT

#saves TFIDF, SVD matrices, terms that appear in the documents to Python "pickle" files
def saveMatricesToPickle(tfidf, terms, sigma, VT):

    with open('./pickle_files/sigma', 'wb') as fp:
        pickle.dump(sigma, fp)

    with open('./pickle_files/vt', 'wb') as fp:
        pickle.dump(VT, fp)

    with open('./pickle_files/tfidf', 'wb') as fp:
        pickle.dump(tfidf, fp)

    with open('./pickle_files/terms', 'wb') as fp:
        pickle.dump(terms, fp)

#removes names, geospatial names etc., takes text with part of speech and ne_chunk tags as argument
def remove_named_entities(text):
    for word in text[:]:
        if hasattr(word, 'label'):
            if word.label() == "NE":
                text.remove(word)
    return text

#filter out part of speech tags and convert words to lower case
#TODO: this could probably by done using zip function
def remove_pos_tags(text):

    words = []
    for word in text:
        words.append(word[0].lower())

    return words