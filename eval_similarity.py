#####################
# Calculates cosine similarity between two terms present in the array of unique
# terms gathered from BNC, returns cosine distance (float -1 to 1)
######################

import numpy
import scipy
import pickle

class Cosine_Similarity:

    def __init__(self):
        self.tfidf, self.terms, self.sigma, self.VT = self._load_matrices();

    def calculate_term_similarity(self, term1, term2):

        if (term1 in self.terms) and (term2 in self.terms):
            index0 = self.terms.index(term1)
            index1 = self.terms.index(term2)
        else:
            return 0

        t0 = numpy.transpose(self.tfidf[index0])
        t1 = numpy.transpose(self.tfidf[index1])

        cos_dist = self._calculate_cosine_similarity(t0, t1)
        print(term1 + " and " + term2 + " similarity:")
        print(1 - cos_dist)

        return (1-cos_dist)


    # convert sigma from an array format to diagonal matrix
    def _create_diagonal_matrix(self):

        diagMatrix = numpy.zeros((len(self.sigma), len(self.sigma)), int)
        numpy.fill_diagonal(diagMatrix, 1)
        diagMatrix = diagMatrix * self.sigma
        return diagMatrix

    # calculate cosine similarity between two terms, where t0 and t1 are pseudo term vectors
    def _calculate_cosine_similarity(self, t0, t1):

        #sigma is in condensed format, expand it to a diagonal matrix
        sigma_diagonal = self._create_diagonal_matrix()

        #invert the matrix
        sigma_inv = numpy.linalg.inv(sigma_diagonal)

        #calculate pseudo term vector
        term_vector0 = numpy.mat(sigma_inv) * numpy.mat(self.VT) * t0
        term_vector1 = numpy.mat(sigma_inv) * numpy.mat(self.VT) * t1

        distance = scipy.spatial.distance.cosine(term_vector0,term_vector1)

        return distance

    @staticmethod
    def _load_matrices():

        with open('./pickle_files/sigma', 'rb') as fp:
            sigma = pickle.load(fp)
        with open('./pickle_files/vt', 'rb') as fp:
            VT = pickle.load(fp)
        with open('./pickle_files/tfidf', 'rb') as fp:
            tfidf = pickle.load(fp)
        with open('./pickle_files/terms', 'rb') as fp:
            terms = pickle.load(fp)

        return tfidf, terms, sigma, VT