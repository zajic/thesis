'''
Load BNC, preprocess and create latent semantic space
'''

# import bnc_lsa
#
# bnc_preprocesser = bnc_lsa.BNCLatentSemanticAnalysis()
# bnc_preprocesser.perform_lsa()

##################################################

'''
Generate extended emotion dictionary
'''
import emotion_vocabulary_builder as eab

vocab_builder = eab.vocabBuilder()
vocab_builder.build_emotion_vocab()



'''
Load news from Event Registry related to '2016 Summer Olympics' concept.
Query articles by '2016 Summer Olympics' published in The Independent.
Query events related to North Korea.
'''

# import load_news
#
# loader = load_news.newsLoader()
# loader.query_articles_by_event("eng-3310179","north korea")
# loader.query_articles_by_concept_plus_media("2016 Summer Olympics","The Independent")
# loader.query_events_by_concept("North Korea")
#

'''
Evalute emotion in articles in a pickle file, reduce dimensionality by PCA and display in a mpld3 plot
'''
# import eval_articles_emotion as eae
#
# emotion_eval = eae.articleEmotionEvaluator('articles_north_korea_preprocessed')
# emotion_eval.pca_2D_visualization()


'''
Create a plot with 20 most frequent words in the BNC
'''
# import word_statistics
#
# stats_generator = word_statistics.wordStatsGenerator()
# stats_generator.most_frequent_bnc('frequent_words_plot',display_labels=True,limit=20)



