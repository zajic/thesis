#####################
# Module identifies emotions in a set of articles from eventregistry.org and visualized them in a 2D or 3D plot.
# Each article is assigned a dictionary of 32 different emotions. In order to visualize the articles,
# dimensionality is reduced either by PCA or multidimensional scaling.
# The expected file is an array of articles (tokenized and preprocessed), each article must have the following properties:
# 'title','source','body'.
# The output is either a matplotlib plot or mpld3 plot (open in a browser).
######################

import pickle
from WNAffect_modules import wnaffect
from collections import Counter
from copy import deepcopy
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import collections
import mpld3
from sklearn.manifold import MDS
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import operator
import matplotlib.patches as mpatches

class articleEmotionEvaluator:

    def __init__(self,filename):

        #set wordnet directories required by wnaffect module
        self.wordnet16_dir = "./wordnet/wordnet-1.6"
        self.wn_domains_dir = "./wordnet/wn-domains-3.2"

        # initialize Wordnet Affect object for emotion tagging
        self.wna = wnaffect.WNAffect(self.wordnet16_dir, self.wn_domains_dir)

        #load articles
        with open('./pickle_files/' + filename, 'rb') as fp:
            self.articles = pickle.load(fp)

        #load emotion dictionary
        with open('./pickle_files/emotion_dictionary_bnc_baby_avg', 'rb') as fp:
               self.em_dict = pickle.load(fp)

        self.coordinates = self.eval_articles_emotions()

    #calculate emotion score for each article in the file provided
    def eval_articles_emotions(self):

        for art in self.articles:
            art['emotion_score'] = {}
            art['emotion_score'], art['emotion_score_sorted'] = self.tag_emotions_alt(art['body'])

        #filter out the coordinates for each article and convert to numpy array
        coordinates = []
        for art in self.articles:
            X = list(art['emotion_score'].values())
            coordinates.append(X)

        return np.array(coordinates)

    '''
    Visualize data in a matplotlib 2D plot, labels displayed on hover.
    Use PCA to reduce dimensionality. 
    '''
    def pca_2D_visualization(self):

        labels = [art['title'] + " (" + art['source'] + ")" for art in self.articles]
        scaled_coordinates = self._pca(self.coordinates,2)
        self._visualize_2D_data(scaled_coordinates, labels, type = 'pca')

    '''
    Visualize data in a mpl_toolkits 2D plot.    
    Use PCA to reduce dimensionality. 
    '''
    def pca_3D_visualization(self):

        labels = [art['title'] + " (" + art['source'] + ")" for art in self.articles]
        scaled_coordinates = self._pca(coordinates, 3)
        visualize_3D_data(coordinates, labels)

    '''
    Scale coordinates with multidimensional scaling and display a plot
    '''
    def mds_2d_visualization(self):

        labels = [art['title'] + " (" + art['source'] + ")" for art in self.articles]
        scaled_coordinates = self._multidimensional_scaling(self.coordinates, 2)
        self._visualize_2D_data(scaled_coordinates, labels, type="mds")

    '''
    Compare PCA and MDS methods, plots are color coded with rainbow scale
    '''
    def compare_pca_mds_rainbow(self):

        sc_coordinates = self._pca(self.coordinates,2)
        colors = self._get_rainbow_color_codes(sc_coordinates)
        scaled_coordinates = self._multidimensional_scaling(self.coordinates, 2)
        # self._visualize_2D_with_color_coding(sc_coordinates, colors, type='pca')
        self._visualize_2D_with_color_coding(scaled_coordinates, colors, type='mds metric')

    '''
    An alternative method for emotion tagging: only words with emotion score higher than 0.3
    (or a threshold of your choice) are tagged with an emotion
    '''
    def tag_emotions_alt(self,article):

        # filter out words that have emotion scores associated
        bag_of_emotion_words = {}
        for word in article:
            #if a word is found in the emotion dictionary
            if word in self.em_dict.keys():
                print(word + str(self.em_dict[word]))
                #copy the entry in the emotion dictionary and add it to the bag of emotion words
                bag_of_emotion_words[word] = deepcopy(self.em_dict[word])

        #set all emotion scores lower than 0.2 to 0 (including negative)
        for word in bag_of_emotion_words:
            for key, value in bag_of_emotion_words[word].items():
                if bag_of_emotion_words[word][key] < 0.2:
                    bag_of_emotion_words[word][key] = 0

        # calculate word frequency
        counts = Counter(article)

        # if a word appears multiple times in the article multiply the emotion scores of a word by the frequency count
        for word in bag_of_emotion_words:
            if counts[word] > 1:
                for key, value in bag_of_emotion_words[word].items():
                    bag_of_emotion_words[word][key] = value * counts[word]

        # empty template for emotion scores
        summed_em = {'apathy': 0, 'neutral-unconcern': 0, 'thing': 0, 'pensiveness': 0, 'gravity': 0,
                     'ambiguous-fear': 0, 'ambiguous-expectation': 0, 'ambiguous-agitation': 0,
                     'surprise': 0, 'gratitude': 0, 'levity': 0, 'positive-fear': 0, 'fearlessness': 0,
                     'positive-expectation': 0, 'self-pride': 0, 'affection': 0, 'enthusiasm': 0,
                     'positive-hope': 0, 'calmness': 0, 'love': 0, 'joy': 0, 'liking': 0, 'humility': 0,
                     'compassion': 0, 'despair': 0, 'shame': 0, 'anxiety': 0, 'negative-fear': 0,
                     'general-dislike': 0, 'sadness': 0, 'daze': 0, 'ingratitude': 0}

        # sum emotion scores for all words in the article
        for item in bag_of_emotion_words:
            summed_em = {k: summed_em.get(k, 0) + bag_of_emotion_words[item].get(k, 0) for k in
                         set(summed_em) | set(bag_of_emotion_words[item])}


        # order emotion keys (apathy etc.) alphabetically
        summed_em_dict = collections.OrderedDict(sorted(summed_em.items()))

        #divide emotion scores by number of emotion words present in the article
        for em in summed_em_dict:
            summed_em_dict[em] = summed_em_dict[em]/len(bag_of_emotion_words)

        sorted_x = sorted(summed_em_dict.items(), key=operator.itemgetter(1), reverse=True)
        # print(sorted_x)

        return summed_em_dict, sorted_x

    '''
    Iterate over the words in an article and if a word is found in the emotion dictionary assign
    corresponding scores to it
    '''
    def tag_emotions(self,article):

        #filter out words that have emotion scores associates
        bag_of_emotion_words = {}
        for word in article:
            #if word found in emotion dictionary
            if word in self.em_dict.keys():
                #copy the corresponding record in the emotion dictionary and add it to the bag of emotion words
                bag_of_emotion_words[word] = deepcopy(self.em_dict[word])

        #calculate word frequency
        counts = Counter(article)

        #if a word appears multiple times in the article multiply the emotion scores of a word by the frequency count
        for word in bag_of_emotion_words:
            if counts[word] > 1:
                for key,value in bag_of_emotion_words[word].items():
                    bag_of_emotion_words[word][key] = value*counts[word]

        print(bag_of_emotion_words)

        # empty template for emotion scores
        summed_em = {'apathy':0,'neutral-unconcern':0,'thing':0,'pensiveness':0,'gravity':0,'ambiguous-fear':0,'ambiguous-expectation':0,'ambiguous-agitation':0,
                     'surprise':0,'gratitude':0,'levity':0,'positive-fear':0,'fearlessness':0,'positive-expectation':0,'self-pride':0,'affection':0,'enthusiasm':0,
                     'positive-hope':0,'calmness':0,'love':0,'joy':0,'liking':0,'humility':0,'compassion':0,'despair':0,'shame':0,'anxiety':0,'negative-fear':0,
                     'general-dislike':0,'sadness':0,'daze':0,'ingratitude':0}

        # sum emotion scores for all words in the article
        for item in bag_of_emotion_words:
            summed_em = {k: summed_em.get(k, 0) + bag_of_emotion_words[item].get(k, 0) for k in set(summed_em) | set(bag_of_emotion_words[item])}

        # order emotion keys (apathy etc.) alphabetically
        summed_em_dict = collections.OrderedDict(sorted(summed_em.items()))

        #divide emotion scores by number of emotion words present in the article
        for em in summed_em_dict:
            summed_em_dict[em] = summed_em_dict[em]/len(bag_of_emotion_words)

        sorted_x = sorted(summed_em_dict.items(), key=operator.itemgetter(1), reverse=True)

        return summed_em_dict, sorted_x

    '''
    reduce dimensionality with PCA
    no_of_components: integer representing the desired number of dimensions
    '''
    @staticmethod
    def _pca(coordinates, no_of_components):

        pca = PCA(n_components=no_of_components, random_state = 123)
        pca.fit(coordinates)
        X = pca.transform(coordinates)

        print("explained variance ratio:")
        print(pca.explained_variance_ratio_)
        return X

    '''
   reduce dimensionality with MDS
   no_of_components: integer representing the desired number of dimensions
    '''
    @staticmethod
    def _multidimensional_scaling(coordinates, no_components):
        print("Applying multidimensional scaling...")
        mds = MDS(n_components=no_components, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=1,
            random_state=123, dissimilarity='euclidean')
        scaled_coordinates = mds.fit_transform(coordinates)

        return scaled_coordinates


    '''
    Used for visualizing points with multiple colors in one plot (matplotlib)
    Legend is commented out.
    Set plt.axis size if you want your data in a square plot.
    '''
    @staticmethod
    def _visualize_2D_with_color_coding(X, colors, type=None):

        print("Creating plot...")
        for index in range(0, len(X)):
            plt.scatter(X[index, 0], X[index, 1], picker=True, color = colors[index], s=10, alpha = 0.5)

        # blue_patch = mpatches.Patch(color='#4286f4', label='The Telegraph')
        # red_patch = mpatches.Patch(color='#f44441', label='The Independent')
        # blue_patch = mpatches.Patch(color='#4286f4', label='Daily Mail')
        # orange_patch = mpatches.Patch(color='#f47616', label='Daily Mail')
        # red_patch = mpatches.Patch(color='#f44441', label='The Guardian')
        # green_patch = mpatches.Patch(color='#2cb240', label='The Guardian')
        plt.legend(handles=[blue_patch, orange_patch])

        #uncomment for a square plot
        # plt.axis([-0.1,0.3,-0.1,0.3])
        # plt.gca().set_aspect('equal', adjustable='box')

        if type == 'pca':

            plt.title("PCA")
            plt.xlabel("PC1")
            plt.ylabel("PC2")

        elif type == 'mds metric':

            plt.title("Metric MDS")
            plt.xlabel("dimension 1")
            plt.ylabel("dimension 2")

        else:
            plt.title("Non-metric MDS")
            plt.xlabel("dimension 1")
            plt.ylabel("dimension 2")


        plt.show()

    '''
    Visualize data points with different colors (each article must have an attribute 'color' assigned).
    '''
    def visualize_2D_data_comparison_pyplot(self):

        labels = [art['title'] + " (" + art['source'] + ")" for art in self.articles]
        scaled_coordinates = self._pca(self.coordinates, 2)
        colors = [x['color'] for x in self.articles]
        self._visualize_2D_with_color_coding(scaled_coordinates, colors, type='pca')


    #reference: https://mpld3.github.io/examples/html_tooltips.html
    '''
    display plot with legend on hover using mpld3 package in a browser window
    '''
    @staticmethod
    def _visualize_2D_data (X, labels, type=None):

        print("Creating plot (this may take a while)...")

        fig, ax = plt.subplots(figsize=(16, 8))

        points = ax.plot(X[:, 0], X[:, 1],'.',markersize=15, alpha=0.5)

        if type == 'pca':

            plt.title("PCA")
            plt.xlabel("PC1")
            plt.ylabel("PC2")

        else:
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension2')
            ax.set_title('MDS', size=20)

        my_labels = labels

        #set style for labels displayed on hover
        my_css = """
        text.mpld3-text, div.mpld3-tooltip {
          font-family: Arial, Helvetica, sans-serif;
          font-weight: bold;
          color: black;
          opacity: 0.7;
          background-color: #FFFCA6;
          padding: 10px;
        }
        """
        tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels=my_labels, css = my_css)

        mpld3.plugins.connect(fig, tooltip)

        mpld3.show()

    '''
    argument: numpy array coordinates
    assign a color from a rainbow scale to each datapoint
    '''
    @staticmethod
    def _get_rainbow_color_codes(X):

        cmap = cm.gist_ncar
        min_x = (min(X[:,0]))
        max_x = (max(X[:,0]))

        norm = Normalize(vmin=min_x, vmax=max_x)

        colors = []
        for index in range(0, len(X)):
            colors.append(cmap(norm(X[index,0])))
            # plt.scatter(X[index, 0], X[index, 1], picker=True, color=cmap(norm(X[index,0])))

        return colors


    #reference: https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
    '''
    Visualize data in a 3D matplotlib plot with popover next to mouse position.
    Args:
    X (np.array) - array of points, of shape (numPoints, 3)
    '''
    def _visualize_3D_data (X, labels):

        fig = plt.figure(figsize = (16,10))
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], picker = True)

        def distance(point, event):
            """Return distance between mouse position and given data point

            Args:
                point (np.array): np.array of shape (3,), with x,y,z in data coords
                event (MouseEvent): mouse event (which contains mouse position in .x and .xdata)
            Returns:
                distance (np.float64): distance (in screen coords) between mouse pos and data point
            """
            assert point.shape == (3,), "distance: point.shape is wrong: %s, must be (3,)" % point.shape

            # Project 3d data space to 2d data space
            x2, y2, _ = proj3d.proj_transform(point[0], point[1], point[2], plt.gca().get_proj())
            # Convert 2d data space to 2d screen space
            x3, y3 = ax.transData.transform((x2, y2))

            return np.sqrt ((x3 - event.x)**2 + (y3 - event.y)**2)

        def calcClosestDatapoint(X, event):
            """"Calculate which data point is closest to the mouse position.

            Args:
                X (np.array) - array of points, of shape (numPoints, 3)
                event (MouseEvent) - mouse event (containing mouse position)
            Returns:
                smallestIndex (int) - the index (into the array of points X) of the element closest to the mouse position
            """
            distances = [distance (X[i, 0:3], event) for i in range(X.shape[0])]
            return np.argmin(distances)

        def annotatePlot(X, index):
            """Create popover label in 3d chart
            Args:
                X (np.array) - array of points, of shape (numPoints, 3)
                index (int) - index (into points array X) of item which should be printed
            """
            # If we have previously displayed another label, remove it first
            if hasattr(annotatePlot, 'label'):
                annotatePlot.label.remove()
            # Get data point from array of points X, at position index
            x2, y2, _ = proj3d.proj_transform(X[index, 0], X[index, 1], X[index, 2], ax.get_proj())
            annotatePlot.label = plt.annotate( labels[index],
                xy = (x2, y2), xytext = (-20, 20), textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
            fig.canvas.draw()


        def onMouseMotion(event):
            """Event that is triggered when mouse is moved. Shows text annotation over data point closest to mouse."""
            closestIndex = calcClosestDatapoint(X, event)
            annotatePlot (X, closestIndex)

        fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)  # on mouse motion

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title("")

        plt.show()