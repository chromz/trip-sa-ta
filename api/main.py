from typing import List

from fastapi import FastAPI, Query

import joblib
from pprint import pprint
from src.models.train import train
from src.models.predict import predict
from src.models.train_topics import train_topics

app = FastAPI()


@app.post('/train')
async def train_model():
    train()
    train_topics()
    return {'Result': 'model.joblib produced'}

@app.get('/predict')
async def predict_review(sentence: str = Query(..., description='Sentences to process')):
    return predict(sentence)

@app.get('/topics')
async def topics():
    lda_model = joblib.load('models/topics.joblib')
    topics = lda_model.print_topics()
    topics_dicts = {}
    topics_dicts['Amenidades'] = topics[0][1]
    topics_dicts['Alrededores'] = topics[1][1]
    topics_dicts['Atencion en pizzerias'] = topics[2][1]
    topics_dicts['Navegacion'] = topics[3][1]
    topics_dicts['Restaurantes'] = topics[4][1]
    topics_dicts['Bienvenida'] = topics[5][1]
    topics_dicts['Estilo y arte'] = topics[6][1]
    topics_dicts['Sueño'] = topics[7][1]
    topics_dicts['Cena'] = topics[8][1]
    topics_dicts['Desayuno y estadia'] = topics[9][1]
    topics_dicts['Compras extras'] = topics[10][1]
    topics_dicts['Entrada al hotel'] = topics[11][1]
    topics_dicts['Transporte'] = topics[12][1]
    topics_dicts['Tiendas de alrededor'] = topics[13][1]
    topics_dicts['Actitud y apariencia del personal'] = topics[14][1]
    topics_dicts['Internet'] = topics[15][1]
    topics_dicts['Puntualidad'] = topics[16][1]
    topics_dicts['Servicios de lujo'] = topics[17][1]
    topics_dicts['Llegada a la habitacion'] = topics[18][1]
    topics_dicts['Estadia en el hotel'] = topics[19][1]

    pprint(topics_dicts)
    return topics_dicts
