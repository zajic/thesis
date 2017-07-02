#reference: https://github.com/clemtoy/WNAffect
from WNAffect_modules import wnaffect
from nltk import pos_tag
import pickle
from nltk.corpus import wordnet as wn
from itertools import groupby
from itertools import chain
from collections import Counter
from eval_similarity import Cosine_Similarity
from copy import deepcopy

wordnet16_dir = "./wordnet/wordnet-1.6"
wn_domains_dir = "./wordnet/wn-domains-3.2"

'''
Load all Wordnet synsets, filter out synsets that are tagged with an emotion in Wordnet Affect.
Save to pickle.
'''
def get_all_emotion_words():
    #initialize Wordnet Affect object for emotion tagging
    wna = wnaffect.WNAffect(wordnet16_dir, wn_domains_dir)

    print("Loading all Wordnet synsets...")
    syns = [s for s in wn.all_synsets()]

    #try looking up emotion for each word in Wordnet
    print("Filtering out synsets that have emotion label associated...")
    terms_affective_dict ={}
    for syn in syns:
        word = syn.name()
        word = word[:word.find(".")]
        wn_pos = {'n':'NN','v':'VB','r':'RB','s':'JJ','a':'JJ'}
        pos = wn_pos[syn.pos()]
        em = wna.get_emotion(word, pos)
        if em is not None:
            #print(str(word) + " " + str(em))
            terms_affective_dict[word] = em.get_level(5).name


    print("Number of words in Wordnet tagged with an emotion:")
    print(len(terms_affective_dict))

    #save to file
    with open('./pickle_files/affective_dictionary', 'wb') as fp:
        pickle.dump(terms_affective_dict, fp)

'''
group affective terms by emotions, i.e. from format
[protective:affection, caring:affection,..]
convert to format
{affection:[protective, caring, etc.], apathy:[dreamy, emotionless, etc.]..}
'''
def group_by_emotion_categories(terms):

    groups = []
    terms.sort(key=lambda x: x[1])
    for key, group in groupby(terms,lambda x: x[1]):
        groups.append(list(group))

    for group in groups:
        cat = group[0][1]
        category = cat[:cat.find(":")]
        list_of_words = []
        for item in group:
            list_of_words.append(item)
        # print(cat + ":" + str(list_of_words))

    return groups

'''
filter out words from BNC that have an emotion associated in Wordnet Affect
'''
def filter_emotion_tagged_terms_from_BNC():

    print("Loading terms...")
    with open('./pickle_files/bnc_preprocessed', 'rb') as fp:
        docs = pickle.load(fp);

    terms = list(chain.from_iterable(docs))

    terms = pos_tag(terms)
    print("Terms with tagged part of speech:")
    print(terms)
    print(len(terms))

    wna = wnaffect.WNAffect(wordnet16_dir, wn_domains_dir)
    terms_affective_dict = []
    for term in terms:
        em = wna.get_emotion(term[0], term[1])
        if em is not None:
            terms_affective_dict.append((term[0], em.get_level(5).name))

    for term in terms_affective_dict:
        print(term)
    print(len(terms_affective_dict))

    with open('./pickle_files/emotion_terms_bnc', 'wb') as fp:
        pickle.dump(terms_affective_dict, fp)

def get_most_frequent_affective_terms(number_of_terms):

    with open('./pickle_files/emotion_terms_bnc', 'rb') as fp:
        terms = pickle.load(fp)

    counts = Counter(terms)
    most_common = counts.most_common(number_of_terms)

    return most_common

def get_most_frequent_terms_bnc(number_of_terms):

    with open('./pickle_files/bnc_preprocessed', 'rb') as fp:
        docs = pickle.load(fp)

    words = list(chain.from_iterable(docs))
    counts = Counter(words)
    most_common = counts.most_common(number_of_terms)

    return most_common


def build_generic_words_vocab():

    #get most frequent 1000 words from BNC
    frequent_bnc_terms = get_most_frequent_terms_bnc(1000)
    bnc_labels, values = zip(*frequent_bnc_terms)
    print(bnc_labels)

    #get 100 most frequent affective words used in BNC
    freq_affective_terms = get_most_frequent_affective_terms(100)
    # affect_labels, affect_values = zip(*freq_affective_terms)
    print(freq_affective_terms)

    cos_sim = Cosine_Similarity()
    similarity_vocab = []
    for generic_word in bnc_labels:
        for affective_word in freq_affective_terms:
            if generic_word != affective_word[0][0]:
                cos_similarity = cos_sim.calculate_term_similarity(generic_word,affective_word[0][0])
                similarity_vocab.append([generic_word,affective_word[0],cos_similarity])

    print(similarity_vocab)

    with open('./pickle_files/similarity_vocab', 'wb') as fp:
        pickle.dump(similarity_vocab, fp)

    vocab = group_vocabulary_by_term(similarity_vocab)


#from an array of tuples creates a dictionary with terms as keys
#example:
def group_vocabulary_by_term(vocab):

    with open('./pickle_files/affective_dictionary', 'rb') as fp:
        dict = pickle.load(fp)

    groups = []
    vocab.sort(key=lambda x: x[0])
    for key, group in groupby(vocab, lambda x: x[0]):
        groups.append(list(group))

    words_with_emotions = {}
    for group in groups:
        cat = group[0][0]
        emotion_scores = []
        for item in group:
            emotion_scores.append([item[1],item[2]])
        words_with_emotions[cat] = emotion_scores

    # for item in words_with_emotions:
    #     print(item + ":" + str(words_with_emotions[item]))

    return words_with_emotions

def calculate_emotion_scores():

    with open('./pickle_files/similarity_vocab', 'rb') as fp:
        vocab = pickle.load(fp)
        words_with_emotions = group_vocabulary_by_term(vocab)

    with open('./pickle_files/affective_dictionary', 'rb') as fp:
        affective_dict = pickle.load(fp)

    # for word in words_with_emotions:
    #     print(words_with_emotions[word])

    em_scores = {'apathy':0,'neutral-unconcern':0,'thing':0,'pensiveness':0,'gravity':0,'ambiguous-fear':0,'ambiguous-expectation':0,'ambiguous-agitation':0,
                 'surprise':0,'gratitude':0,'levity':0,'positive-fear':0,'fearlessness':0,'positive-expectation':0,'self-pride':0,'affection':0,'enthusiasm':0,
                 'positive-hope':0,'calmness':0,'love':0,'joy':0,'liking':0,'humility':0,'compassion':0,'despair':0,'shame':0,'anxiety':0,'negative-fear':0,
                 'general-dislike':0,'sadness':0}

    em_dict = {}
    #assign max similarity score for each word-emotion pair
    for word in words_with_emotions:
        word_em_scores = deepcopy(em_scores)
        # if the word is a Wordnet affect word itself assign score 1 to the emotion the word represents (e.g. love:['love':1, 'gravity':0.48 etc.]
        if word in affective_dict.keys():
            word_em_scores[affective_dict[word]] = 1
        for em in words_with_emotions[word]:
            if (word_em_scores[em[0][1]] == 0) or (word_em_scores[em[0][1]] < em[1]):
                word_em_scores[em[0][1]] = em[1]

        em_dict[word] = word_em_scores

    for item in em_dict:
        print(item + str(em_dict[item]))


    with open('./pickle_files/emotion_dictionary', 'wb') as fp:
        pickle.dump(em_dict, fp)


calculate_emotion_scores()
# build_generic_words_vocab()
# get_all_emotion_words()
# filter_emotion_tagged_terms_from_BNC()
