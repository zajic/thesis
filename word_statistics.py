#reference: https://stackoverflow.com/questions/35596128/how-to-generate-a-word-frequency-histogram-where-bars-are-ordered-according-to
#####################
#
# Creates histograms of most frequent terms contained in the BNC sample
# and the most frequent affective terms used in the BNC sample
#
######################

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools

class wordStatsGenerator:


    # display_labels: display bar labels (set to False when the number of bars is too high)
    # limit: number of most frequent words you wish to display
    @staticmethod
    def most_frequent_bnc(output_filename, display_labels = False, limit = 100):

        with open('./pickle_files/bnc_baby_preprocessed', 'rb') as fp:
            docs = pickle.load(fp)

        # join arrays of each doc to one array
        words = list(itertools.chain.from_iterable(docs))
        print(words)
        print(len(words))

        counts = Counter(words)
        most_common = counts.most_common(limit)
        print(most_common)

        #split tuples to two arrays
        labels, values = zip(*most_common)

        # sort values in descending order
        indSort = np.argsort(values)[::-1]

        # rearrange data
        labels = np.array(labels)[indSort]
        values = np.array(values)[indSort]

        indexes = np.arange(len(labels))
        plt.bar(indexes, values)
        #color = [0.973, 0.808, 0.8]

        # add labels
        plt.ylabel("frequency")

        #display term for each bar, otherwise numerical label will be displayed
        if display_labels:
            plt.xticks(indexes, labels)
            plt.xticks(rotation=45,fontsize=10)
            plt.xlabel("term", fontsize = 15)

        else:
            plt.xlabel("word frequency ranking (1st most frequent, 2nd etc.)", fontsize=10)

        output_file = './plots/' + output_filename + '.pdf'
        plt.show()

    # display_labels: display bar labels (set to False when the number of bars is too high)
    # limit: number of most frequent words you wish to display
    @staticmethod
    def most_frequent_affective(output_filename,display_labels = False,limit=100):

        with open('./pickle_files/emotion_terms_bnc_baby', 'rb') as fp:
            terms = pickle.load(fp)

        words = [w[0] for w in terms]

        counts = Counter(words)
        most_common = counts.most_common(limit)
        print(most_common)
        labels, values = zip(*most_common)

        # sort values in descending order
        indSort = np.argsort(values)[::-1]

        # rearrange your data
        labels = np.array(labels)[indSort]
        values = np.array(values)[indSort]

        indexes = np.arange(len(labels))

        plt.bar(indexes, values)

        # display term label for each bar, otherwise numerical label will be displayed
        if display_labels:
            plt.xticks(indexes, labels)
            plt.xticks(rotation=60, fontsize=10)
            plt.xlabel("emotion word", fontsize=9)

        else:
            plt.xlabel("word frequency ranking (1st most frequent, 2nd etc.)", fontsize=10)


        output_file = './plots/' + output_filename + '.pdf'
        plt.savefig(output_file)
        plt.show()

    @staticmethod
    def frequent_words(words, limit = 100):

        counts = Counter(words)
        most_common = counts.most_common(limit)
        print(most_common)
        labels, values = zip(*most_common)
        # sort values in descending order
        indSort = np.argsort(values)[::-1]

        # rearrange data
        labels = np.array(labels)[indSort]
        values = np.array(values)[indSort]

        indexes = np.arange(len(labels))

        plt.bar(indexes, values)


        # display term label for each bar, otherwise numerical label will be displayed
        plt.xticks(indexes, labels)
        plt.xticks(rotation=60, fontsize=10)
        plt.xlabel("word", fontsize=9)

        plt.show()



