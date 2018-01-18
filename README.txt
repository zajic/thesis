Note: Event registry API cannot be accessed from University of St Andrews network as it is considered a phishing website.

######################
#
# Required files - BNC and Wordnet were not submitted with the code due to their enormous size
#
######################

1) British National Corpus (BNC) or BNC Baby, to be downloaded from Oxford text archive and placed in the directory "corpora".
Note that the nltk.corpus.reader.BNCCorpusReader expects the xml files to be nested within two directories, which is the case
for BNC but not for BNC Baby. For BNC Baby the path structure has to be updated manually, so for example if you set the path to BNC:

self.path_to_bnc = './corpora/bnc_baby'

you need to create two sub-directories A and A0 and place your files there:

".\corpora\bnc_baby\A\A0\A0.xml"


2) WordNet 1.6 (UNIX-like) and WordNet-Domains 3.2., to be obtained from http://wndomains.fbk.eu/download.html and placed in "wordnet" directory

3) wnaffect.py and emotion.py external modules to be downloaded from https://github.com/clemtoy/WNAffect and placed in "WNAffect_modules" directory.

########################
#
# Required packages
#
########################
collections
copy
eventregistry
itertools
math
matplotlib
mpl_toolkits
mpld3
nltk
numpy
operator
pandas
pickle
scipy
sklearn
string
wnaffect (see above)


#####################
#
# Usage examples
#
#####################

1) Load BNC and convert it to latent semantic space
============================================================

import bnc_lsa

bnc_preprocesser = bnc_lsa.BNCLatentSemanticAnalysis()
bnc_preprocesser.perform_lsa()

2) generate extended emotion lexicon
===========================================

import emotion_vocabulary_builder as eab

vocab_builder = eab.vocabBuilder()
vocab_builder.build_emotion_vocab()

3) download news articles
========================================

import load_news

loader = load_news.newsLoader()
loader.query_events_by_concept("2016 Summer Olympics")

to see the articles "unpickle" the file in a console:

with open('./pickle_files/' + $yourFilename, 'rb') as fp:
		articles = pickle.load(fp);
		
...

For more examples see examples.py.


####################
#
# FAQ:
#
####################

1) Python complains about unicode characters:
BNC contains some foreign words that have unicode characters in them. There are also unicode chars in the code itself.
Add the following to the top of the code

# -*- coding: utf-8 -*-

If that still does not work, add w.encode('utf-8') for every variable w (word) in bnc_lsa.py when iterating over word in the BNC.

Alternatively, run the code with Python 3.

2) Event registry cannot be accessed, no results are returned for my queries:
You need to connect outside of St Andrews University network.
Alternatively, you may have exceeded the limit 2000 queries per day.

3) I have an old version of Python on a school machine but want to run your code:

You can create a Python virtual environment with a higher version of Python

python3.6 -m virtualenv <my-env-name>
source  <my-vitrual-env-name>/bin/activate




