# Sentiment Analysis CIL

# Preparations
To be able to train and evaluate the fastText model, you are required to install the following python modules via pip.
```
numpy
pickle
sklearn
tensorflow
fastText
nltk
keras
xgboost
pandas
```

# How to run

To run our ensemble on the small dataset, simply execute our main script:
```
python3 main.py
```

If data from the models are already available data generation can be skipped:
```
python3 main.py --skipBuild --skipProbs
```
Optionally, the file paths can be specified with --trainNegPath,
--trainPosPath, --testPath, --submPath and --probsPath.


All models (except the ensemble itself) can be validated:
```
python3 main.py --validateAll
```

And submission files for all models generated using:
```
python3 main.py --submitAll
```

# Structure

## Models
We use XGBoost ensemble to combine multiple models.
Every model has its own file in the models folder and all models inherit from the model baseclass for a common interface.

## Data
This repo only contains the small dataset to get everything running.
Link to full data: https://polybox.ethz.ch/index.php/s/6P4al3Rb5CZM5Pw
This link not only contains the full data sets of tweets but also precomputed embeddings and probabilities for relevant models in order to make also stepwise reconstruction of the submission possible

## Experiments

Experimental code is located in the experiments folder, each with their own readme for installation/run instructions.
