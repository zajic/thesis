import pickle
from WNAffect_modules import wnaffect
from collections import Counter


wordnet16_dir = "./wordnet/wordnet-1.6"
wn_domains_dir = "./wordnet/wn-domains-3.2"

# initialize Wordnet Affect object for emotion tagging
wna = wnaffect.WNAffect(wordnet16_dir, wn_domains_dir)

def main():

    article = load_article('chance_for_europe_art2')
    dict = load_emotion_dict()
    tag_emotions(article,dict)


def load_article(article_name):

    path = './pickle_files/' + article_name

    with open(path, 'rb') as fp:
           article = pickle.load(fp)

    print(article)

    # tag part of speech
    # pos_tagged_article = []
    # for par in article:
    #     par = pos_tag(par)
    #     pos_tagged_article.append(par)

    return article

def load_emotion_dict():

    with open('./pickle_files/emotion_dictionary', 'rb') as fp:
           em_dict = pickle.load(fp)

    print(em_dict)
    return em_dict

def test():

    with open('./pickle_files/affective_dictionary', 'rb') as fp:
        dict = pickle.load(fp)

    with open('./pickle_files/emotion_dictionary', 'rb') as fp:
        em_dict = pickle.load(fp)

    for item in dict:
        if item in em_dict.keys():
            print(item)

    print(em_dict['expect'])
        # print(dict)
        # print(em_dict)

def tag_emotions(article,em_dict):

    #filter out words that have emotion scores associates
    bag_of_emotion_words = {}
    for word in article:
        if word in em_dict.keys():
            bag_of_emotion_words[word] = em_dict[word]

    #calculate word frequency
    counts = Counter(article)

    #if a word appears multiple times in the article multiply the emotion scores of a word by the frequency count
    for word in bag_of_emotion_words:
        if counts[word] > 1:
            for key,value in bag_of_emotion_words[word].items():
                bag_of_emotion_words[word][key] = value*counts[word]

    #sum all emotion scores (divide by number of emotion words?)
    summed_em = Counter({})
    for item in bag_of_emotion_words:
        add_dict = Counter(bag_of_emotion_words[item])
        summed_em += add_dict

    for item in summed_em:
        summed_em[item] /= len(bag_of_emotion_words)

    for em in summed_em:
        print(em + " " + str(summed_em[em]))


main()