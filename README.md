# Process

## Build Classifiers
- Write new classifier that is scikit-learn compatible in src/classifier.
- Train the classifier with src/train_models.py and store it in model/

## Run Server
- For releasing, execute src/server.py
- For debugging, execute src/server.py -d

## Test Client& Server connection
- For releasing, execute src/client_test.py
- For releasing, execute src/client_test.py -d

# Server & Client Protocol
1. (GET) /litsmodel

```json
Input:
None
Output:
{'model1_name':['emotion_A', 'emotion_B', 'emotion_C'], 'model2_name':['emotion_D', 'emotion_E']}
```

2. (POST) /predict
```json
Input:
{'model':'model1_name', 'text': 'whatever text here'}
Output:
{"res": [0.2, 0.3, 0.9]} //The probability (or something indicating the score of prediction) for each emotion belonging to the queried model name

```

3. (POST) /log
```json
Input:
{//whatever json}
Output:
None
```



