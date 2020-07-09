# trip-sa-ta


# Dev execution
```bash
uvicorn api.main:app --reload
```

# To train topic modeling and naive bayes prediction (just hit the endpoint train)

```bash
curl -X POST 'localhost:8000/train'
```

# To predict if a review is positive or negative

```bash
curl 'localhost:8000/predict?sentences=this+house+sucks+terrible+bad+shower+weather+waitress'
```

# To show topics
```
curl 'localhost:8000/topics'
```

