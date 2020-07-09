import joblib
from typing import List

import numpy as np
from src.features.tokenize import tokenize_classes

def predict(document):
    classes = {0: [document]}
    model = joblib.load('models/model.joblib')
    doc = tokenize_classes(classes, load_bigrams=True)[0][0]

    none = 1 ** -24
    log_pos_prob = model['POS_PROB']
    log_neg_prob = model['NEG_PROB']
    for word in doc:
        if word in model['COND_POS_PROBS']:
            log_pos_prob += model['COND_POS_PROBS'][word]
        else:
            log_pos_prob += none
        if word in model['COND_NEG_PROBS']:
            log_neg_prob += model['COND_NEG_PROBS'][word]
        else:
            log_neg_prob += none
    veredict = None
    prob = None
    if log_pos_prob > log_neg_prob:
        veredict = 'POS'
        prob = np.exp(log_pos_prob)
    else:
        veredict = 'NEG'
        prob = np.exp(log_neg_prob)

    print(log_pos_prob, log_neg_prob)
    return {
        'veredict': 'POS' if log_pos_prob > log_neg_prob else 'NEG',
        'prob': prob,
    }


