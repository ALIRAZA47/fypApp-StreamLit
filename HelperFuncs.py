import pandas as pd


def generateConfusionMatrix(data):
        confMatrix = pd.crosstab(data['True Sentiment'], data['Predicted Sentiment'])
