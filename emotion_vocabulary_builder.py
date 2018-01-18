#reference: WNAffect_modules by https://github.com/clemtoy/WNAffect
from WNAffect_modules import wnaffect
from nltk import pos_tag
import pickle
from nltk.corpus import wordnet as wn
from itertools import groupby
from itertools import chain
from collections import Counter
from eval_similarity import Cosine_Similarity
from copy import deepcopy

#set path to wordnet and wordnet domains file
wordnet16_dir = "./wordnet/wordnet-1.6"
wn_domains_dir = "./wordnet/wn-domains-3.2"

#set the limit of words that will be compared
GENERIC_WORDS_LIMIT = 1000
AFFECTIVE_WORDS_LIMIT = 500

class vocabBuilder:

        def build_emotion_vocab(self):

            self._get_all_wordnet_emotion_words()
            self._filter_emotion_tagged_terms_from_BNC()
            self._build_generic_words_vocab()
            self._calculate_emotion_scores_average()

        '''
        Load all Wordnet synsets, filter out synsets that are tagged with an emotion in Wordnet Affect.
        Save to pickle.
        '''
        @staticmethod
        def _get_all_wordnet_emotion_words():

            #initialize Wordnet Affect object for emotion tagging
            wna = wnaffect.WNAffect(wordnet16_dir, wn_domains_dir)

            print("Loading all Wordnet synsets...")
            syns = [s for s in wn.all_synsets()]

            #try looking up emotion for each word in Wordnet
            print("Filtering out synsets that have emotion label associated...")
            terms_affective_dict ={}
            for syn in syns:
                #get the synset name
                word = syn.name()
                #trim it
                word = word[:word.find(".")]
                #map wordnet part of speech tags to part of speech tags expected by the wnaffect module
                wn_pos = {'n':'NN','v':'VB','r':'RB','s':'JJ','a':'JJ'}
                pos = wn_pos[syn.pos()]
                #get emotion of a given word
                em = wna.get_emotion(word, pos)
                #if emotion found for a give word, add it to the affective dictionary
                if em is not None:
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
        @staticmethod
        def _group_by_emotion_categories(terms):

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

            return groups

        '''
        filter out words from BNC that have an emotion associated in Wordnet Affect
        '''
        @staticmethod
        def _filter_emotion_tagged_terms_from_BNC():

            print("Loading terms...")
            with open('./pickle_files/bnc_baby_preprocessed', 'rb') as fp:
                docs = pickle.load(fp);

            #concatenate BNC documents (arrays) to one array
            terms = list(chain.from_iterable(docs))

            #tag part of speech
            terms = pos_tag(terms)
            print("Terms with tagged part of speech:")
            print(terms)
            print(len(terms))

            #look up terms from BNC in Wordnet Affect and append to an array of affective terms as tuples (term,emotion)
            wna = wnaffect.WNAffect(wordnet16_dir, wn_domains_dir)
            terms_affective_dict = []
            for term in terms:
                em = wna.get_emotion(term[0], term[1])
                if em is not None:
                    terms_affective_dict.append((term[0], em.get_level(5).name))

            with open('./pickle_files/emotion_terms_bnc_baby', 'wb') as fp:
                pickle.dump(terms_affective_dict, fp)

        @staticmethod
        def _get_most_frequent_affective_terms(number_of_terms):

            print("Getting most frequent affective terms from BNC...")

            with open('./pickle_files/emotion_terms_bnc_baby', 'rb') as fp:
                terms = pickle.load(fp)

            #calculate frequencies of each term
            counts = Counter(terms)

            #get x most frequent terms
            most_common = counts.most_common(number_of_terms)

            return most_common

        @staticmethod
        def _get_most_frequent_terms_bnc(number_of_terms):

            print("Getting most frequent generic terms from BNC...")

            with open('./pickle_files/bnc_baby_preprocessed', 'rb') as fp:
                docs = pickle.load(fp)

            #concatenate bnc documents (arrays) to one array
            words = list(chain.from_iterable(docs))

            # calculate frequencies of each term
            counts = Counter(words)

            # get x most frequent terms
            most_common = counts.most_common(number_of_terms)

            return most_common

        '''
        Calculate cosine similarity for a predefined number of most frequent BNC generic terms
        and a predefined number of affective terms from BNC.
        The result is stored as an array of elements in the format: [term, (affective_term, emotion), cosine_similarity]
        Example: ['search', ('weight', 'sadness'), -0.070528943323980187]
        '''
        def _build_generic_words_vocab(self):

            print("Calculating cosine similarity of generic and emotion terms in BNC...")

            #get most frequent x words from BNC
            frequent_bnc_terms = self._get_most_frequent_terms_bnc(GENERIC_WORDS_LIMIT)
            bnc_labels, values = zip(*frequent_bnc_terms)

            #get x most frequent affective words used in BNC
            freq_affective_terms = self._get_most_frequent_affective_terms(AFFECTIVE_WORDS_LIMIT)

            #instantiate cosine similarity class
            cos_sim = Cosine_Similarity()

            #compare all generic terms to affective terms and append to similarity_vocab
            similarity_vocab = []
            for generic_word in bnc_labels:
                for affective_word in freq_affective_terms:
                    if generic_word != affective_word[0][0]:
                        cos_similarity = cos_sim.calculate_term_similarity(generic_word,affective_word[0][0])
                        similarity_vocab.append([generic_word,affective_word[0],cos_similarity])

            print(similarity_vocab)

            with open('./pickle_files/similarity_vocab_bnc_baby', 'wb') as fp:
                pickle.dump(similarity_vocab, fp)

        '''
          Create a dictionary with terms as keys from an arrray of tuples
        '''
        @staticmethod
        def _group_vocabulary_by_term(vocab):

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

            return words_with_emotions

        '''
         Takes an array of generic/affective term cosine similarity scores in the format:
         [term, (affective_term, emotion), cosine_similarity]
         and calculates the MAXIMUM emotion score for each generic term.
         
         Result stored in a dictionary format:
         {'able': {'apathy': 0, 'neutral-unconcern': 0.089441580165714241, 'thing': 0.39391235181565454,...}
         '''
        def _calculate_emotion_scores_max(self):

            print("Calculating emotion scores of each generic term....")

            with open('./pickle_files/similarity_vocab_bnc_baby', 'rb') as fp:
                vocab = pickle.load(fp)

            words_with_emotions = self._group_vocabulary_by_term(vocab)

            with open('./pickle_files/affective_dictionary', 'rb') as fp:
                affective_dict = pickle.load(fp)

            #empty emotion scores template used for each word
            em_scores = {'apathy':0,'neutral-unconcern':0,'thing':0,'pensiveness':0,'gravity':0,'ambiguous-fear':0,'ambiguous-expectation':0,'ambiguous-agitation':0,
                         'surprise':0,'gratitude':0,'levity':0,'positive-fear':0,'fearlessness':0,'positive-expectation':0,'self-pride':0,'affection':0,'enthusiasm':0,
                         'positive-hope':0,'calmness':0,'love':0,'joy':0,'liking':0,'humility':0,'compassion':0,'despair':0,'shame':0,'anxiety':0,'negative-fear':0,
                         'general-dislike':0,'sadness':0,'daze':0,'ingratitude':0}

            em_dict = {}
            #assign max similarity score for each word-emotion pair
            for word in words_with_emotions:
                print(word)

                word_em_scores = deepcopy(em_scores)
                # if the word is a Wordnet affect word itself assign score 1 to the emotion the word represents (e.g. love:['love':1, 'gravity':0.00 etc.]
                if word in affective_dict.keys():
                    word_em_scores[affective_dict[word]] = 1

                for em in words_with_emotions[word]:
                    if (word_em_scores[em[0][1]] == 0) or (word_em_scores[em[0][1]] < em[1]):
                        word_em_scores[em[0][1]] = em[1]

                em_dict[word] = word_em_scores

            for item in em_dict:
                print(item + str(em_dict[item]))

            print("\n Automatically generated emotion vocabulary saved to 'pickle_files' directory.")
            with open('./pickle_files/emotion_dictionary_bnc_baby_max', 'wb') as fp:
                pickle.dump(em_dict, fp)

        '''
        Takes an array of generic/affective term cosine similarity scores in the format:
        [term, (affective_term, emotion), cosine_similarity]
        and calculates the AVERAGE emotion score for each generic term.

        Result stored in a dictionary format:
        {'able': {'apathy': 0, 'neutral-unconcern': 0.089441580165714241, 'thing': 0.39391235181565454,...}
        '''
        def _calculate_emotion_scores_average(self):

            print("Calculating emotion scores of each generic term....")

            with open('./pickle_files/similarity_vocab_bnc_baby', 'rb') as fp:
                vocab = pickle.load(fp)

            words_with_emotions = self._group_vocabulary_by_term(vocab)

            with open('./pickle_files/affective_dictionary', 'rb') as fp:
                affective_dict = pickle.load(fp)

            # empty emotion scores template used for each word
            em_scores = {'apathy': 0, 'neutral-unconcern': 0, 'thing': 0, 'pensiveness': 0, 'gravity': 0,
                         'ambiguous-fear': 0, 'ambiguous-expectation': 0, 'ambiguous-agitation': 0,
                         'surprise': 0, 'gratitude': 0, 'levity': 0, 'positive-fear': 0, 'fearlessness': 0,
                         'positive-expectation': 0, 'self-pride': 0, 'affection': 0, 'enthusiasm': 0,
                         'positive-hope': 0, 'calmness': 0, 'love': 0, 'joy': 0, 'liking': 0, 'humility': 0,
                         'compassion': 0, 'despair': 0, 'shame': 0, 'anxiety': 0, 'negative-fear': 0,
                         'general-dislike': 0, 'sadness': 0, 'daze': 0, 'ingratitude': 0}

            em_dict = {}
            # assign average similarity score for each word-emotion pair
            for word in words_with_emotions:
                word_em_scores = deepcopy(em_scores)
                #if the word is present in Wordnet affect assign score 1 to the corresponding emotion
                if word in affective_dict.keys():
                    word_em_scores[affective_dict[word]] = 1
                else:
                    #calculate emotion score for a given word and each emotion
                    for em in em_scores:
                        sum_em = 0
                        count = 0
                        print(word)
                        #iterate over affective terms compared to the generic word and sum up their scores
                        for x in words_with_emotions[word]:
                            if x[0][1] == em:
                                print(x[0][0] + ":" + em + ":" + str(x[1]))
                                sum_em += x[1]
                                count += 1
                        #avoid dividing by zero
                        if count != 0:
                            #divide the summed up emotion score by the number of words
                            word_em_scores[em] = sum_em / count
                            print(word_em_scores[em])

                em_dict[word] = word_em_scores

            for item in em_dict:
                print(item + str(em_dict[item]))

            print("\n Automatically generated emotion vocabulary saved to 'pickle_files' directory.")
            with open('./pickle_files/emotion_dictionary_bnc_baby_avg', 'wb') as fp:
                pickle.dump(em_dict, fp)

