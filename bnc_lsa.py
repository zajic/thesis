#####################
# Module loads files from British National Corpus and pre-processes them (tokenize, remove stop words, special characters).
# Creates TFIDF matrix and generates U, sigma and VT (Singular Value Decomposition).
# TFIDF, terms, sigma  and VT are stored as pickle files
######################

import pickle
import numpy
import pandas
from scipy import sparse
from nltk import ne_chunk, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.bnc import BNCCorpusReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.extmath import randomized_svd

class BNCLatentSemanticAnalysis:

    def __init__(self):
        self.path_to_bnc = './corpora/bnc_one_file'
        self.output_file_name = 'bnc_baby_preprocessed'
        self.docs = self._read_BNC(self.path_to_bnc)

    def set_bnc_path(self, path):
        self.path_to_bnc = path

    def set_output_file_name(self, name):
        self.output_file_name = name

    def perform_lsa(self):

        tfidf, terms = self._perform_TFIDF(self.docs)
        #convert tfidf matrix from condensed to dense format and transpose
        tfidf = sparse.csr_matrix.todense(tfidf)
        tfidf = numpy.transpose(tfidf)

        U, sigma, VT = self._singular_value_decomp(tfidf,100)

        self._save_matrices_to_pickle(tfidf,terms,sigma,VT)


    #load and parse files from British National Corpus
    def _read_BNC(self,path_to_bnc):

        print("Reading and processing BNC sample (this may take several minutes)...")

        #load all files in BNC corpus (BNCCorpusReader expects the files to be found within 2 subfolders)
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

            print("\n file" + file)
            # remove special chars, stopwords and append to an array of documents
            preprocessed_text = self._preprocess(text)
            text_corpus.append(preprocessed_text)

            #print document sample
            print(preprocessed_text[:50])
            print("...etc.")

        print("\nnumber of documents in the corpus: " + str(len(text_corpus)))

        # print file names in the corpus
        print("documents: " + str(bnc_reader.fileids()))

        #these files are used for analytical purposes (number of words, most frequent words etc.)
        print("Saving preprocessed BNC text to the current directory...")
        with open('./pickle_files/bnc_baby_preprocessed', 'wb') as fp:
            pickle.dump(text_corpus, fp)

        #save unprocessed (but tokenized) files for further analysis purposes
        with open('./pickle_files/bnc_baby_unprocessed', 'wb') as fp:
            pickle.dump(text_unprocessed, fp)

        return text_corpus

    #remove special characters, stop words, numbers, words shorter than 3 chars, named entities (cities, persons etc.)
    # rermove words that do not exist in Wordnet dictionary, lemmatize nouns (plural to singular) - lemmatizing commented out for now
    def _preprocess(self,text):

        stop_words = self._define_stop_words()
        print("words before preprocessing: " + str(len(text)))

        # remove words shorter than 3 chars
        text = [w for w in text if len(w) > 2]
        #remove stop words
        text = [w for w in text if not w.lower() in stop_words]
        #remove numbers
        text = [w for w in text if not self._has_numbers(w)]
        #remove words containing symbols such as "½"
        text = [w for w in text if not self._contains_unicode_fractions(w)]

        #tag named entities in text and remove them
        text = ne_chunk(pos_tag(text), binary=True)
        text = self._remove_named_entities(text)
        #remove part of speech tags added by ne_chunk
        text = self._remove_pos_tags(text)

        #remove words that do not exist in Wordnet
        final_text = [w for w in text if len(wn.synsets(w))>0]

        print("words after preprocessing: " + str(len(final_text)))

        return final_text

    #https://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
    @staticmethod
    def _has_numbers(word):
        return any(char.isdigit() for char in word)

    #check whether a string contains unicode fractions
    @staticmethod
    def _contains_unicode_fractions(word):

        fractions = ["½","⅓","⅔","¼","¾","⅕","⅖","⅗","⅘","⅙","⅚","⅐","⅛","⅜","⅝","⅞","⅑","⅒"]
        return any(char in fractions for char in word)

    #amend  the list of stopwords with more words to be excluded
    @staticmethod
    def _define_stop_words():

        #TODO: extend this list with words that have definitely no affective meaning
        stop_words = set(stopwords.words('english'))
        months = ["january","february","march","april","may","june","july","august","september","october","november","december"]
        days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
        other = ["yesterday","today","tomorrow"]
        stop_words |= set(days)
        stop_words |= set(months)

        return stop_words

    #returns TFIDF matrix and an array of unique terms in a set of documents, and prints TFIDF matrix as pandas dataframe (with word labels)
    @staticmethod
    def _perform_TFIDF(docs):

        # convert words from an array to plaintext for each doc
        for i in range(len(docs)):
            docs[i] = " ".join(docs[i])

        # initialize the vectorizer
        transformer = TfidfVectorizer()
        tfidf = transformer.fit_transform(docs)

        #convert to panda dataframe (just for neat printing purposes)
        tfidf_panda = pandas.DataFrame(tfidf.toarray().transpose(), index=transformer.get_feature_names())

        print("\nTFIDF as pandas dataframe:")
        print(tfidf_panda)
        terms = transformer.get_feature_names()
        return tfidf, terms

    @staticmethod
    def _singular_value_decomp(tfidf_matrix,components):

        U, sigma, VT = randomized_svd(tfidf_matrix,
                                      n_components=components,
                                      n_iter=20,
                                      random_state=123)

        print("\n singular value decomposition: \n")
        print("\n" + str(U))
        print("\n" + str(sigma))
        print("\n" + str(VT))

        return U,sigma,VT

    #saves TFIDF, SVD matrices, terms that appear in the documents to pickle files
    @staticmethod
    def _save_matrices_to_pickle(tfidf, terms, sigma, VT):

        with open('./pickle_files/sigma', 'wb') as fp:
            pickle.dump(sigma, fp)

        with open('./pickle_files/vt', 'wb') as fp:
            pickle.dump(VT, fp)

        with open('./pickle_files/tfidf', 'wb') as fp:
            pickle.dump(tfidf, fp)

        with open('./pickle_files/terms', 'wb') as fp:
            pickle.dump(terms, fp)

    #removes names, geospatial names etc., takes text tagged with part of speech and ne_chunk tags as argument
    @staticmethod
    def _remove_named_entities(text):
        for word in text[:]:
            if hasattr(word, 'label'):
                if word.label() == "NE":
                    text.remove(word)
        return text

    #filter out part of speech tags and convert words to lower case
    @staticmethod
    def _remove_pos_tags(text):

        words = []
        for word in text:
            words.append(word[0].lower())

        return words