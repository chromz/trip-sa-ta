import enum
from typing import Dict, List
import re

import pandas as pd
from pandas import DataFrame


class Sentiments(enum.Enum):
    POS = 'POS'
    NEG = 'NEG'


def read_sample() -> DataFrame:
    df = pd.read_csv('data/raw/reviews.csv')
    df['rating'] = df['rating'].astype(dtype='int64')

    return df


def create_classes(df: DataFrame) -> Dict[str, List[str]]:
    df['sentiment'] = df['rating'].apply(lambda x: Sentiments.POS if x>=40 else Sentiments.NEG)

    review_classes = {
        sentiment.value: df[df['sentiment'] == sentiment]['review'].values.tolist()
        for sentiment in Sentiments
    }

    return review_classes

def clean_sentences(df):
    data = df.review.values.tolist()
    data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]
    data = [re.sub(r'\s+', ' ', sent) for sent in data]
    data = [re.sub(r"\'", "", sent) for sent in data]
    return data
