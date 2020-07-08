import joblib
import logging
import spacy
import pandas as pd
from src.data.prepare_data import clean_sentences
import gensim.corpora as corpora
from src.features.topic_utils import sent_to_words, remove_stopwords, make_bigrams, lemmatization
from typing import List
from gensim.corpora import Dictionary
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

def _train(corpus: List, dictionary: Dictionary) -> LdaModel:
    lda_model = LdaModel(corpus=corpus,
                         id2word=dictionary,
                         num_topics=20,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)
    return lda_model


def _eval(model: LdaModel, corpus: List, data: List[str], dictionary: Dictionary):
    logging.info(f'Perplexity: {model.log_perplexity(corpus)}')

    # Score de coherencia
    coherence_model_lda = CoherenceModel(model=model,
                                         texts=data,
                                         dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    logging.info(f'Coherence Score: {coherence_lda}')


def train_topics():
    df = pd.read_csv('data/raw/reviews.csv')
    data_words = clean_sentences(df)

    bigram = Phrases(data_words, min_count=5, threshold=100)
    trigram = Phrases(bigram[data_words], threshold=100)

    # Aplicamos el conjunto de bigrams/trigrams a nuestros documentos
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    data_words_nostops = remove_stopwords(data_words)
    data_words_bigrams = make_bigrams(data_words_nostops, bigram_mod)
    nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])
    data_lemmatized = lemmatization(nlp, data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    id2word = Dictionary(data_lemmatized)

    corpus = [id2word.doc2bow(text) for text in data_lemmatized]

    model = _train(corpus, id2word)
    _eval(model, corpus, data_lemmatized, id2word)
    logging.info('Dumping topics model...')
    joblib.dump(model, 'models/topics.joblib')

