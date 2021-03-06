from turtle import width
import pandas as pd
import streamlit as st

# ## Import Libs and Data for Spark

# %%
import pandas as pd
import numpy as np
import re

import json
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline


# %%
# tweets_df = pd.read_csv('tweets.csv')
# tweets_df.columns = ['sentiment','tweet_id','timestamp','q','user','text']
# print(tweets_df)


colnames=[ 'True Sentiment', 'Tweet ID', 'Tweet'] 
# tweets_df.head()

# %%
def readAndShowData(filename="data/retweet_dataset_tweets.csv"):
    tweets_df = pd.read_csv(filename,sep="," , skiprows=1, header=None, names=colnames) 
    tweets_df = tweets_df[(tweets_df.Tweet != "Not Available")]
    tweets_df = tweets_df[(tweets_df.Tweet != "")]
    tweets_df = tweets_df[(tweets_df.Tweet != " ")]

    tweets_df.head()
    return tweets_df


# %% [markdown]
# ## Preparing and Training Model

# %%
# %%
def startSparkAndPreparePipeline():
    spark = sparknlp.start()
    print("Spark NLP version", sparknlp.version())
    print("Apache Spark version:", spark.version)

    documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")
        
    use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")


    sentimentdl = SentimentDLModel.pretrained(name="sentimentdl_use_twitter", lang="en")\
        .setInputCols(["sentence_embeddings"])\
        .setOutputCol("sentiment")

    nlpPipeline = Pipeline(
        stages = [
            documentAssembler,
            use,
            sentimentdl
        ])
    empty_ds = spark.createDataFrame([['']]).toDF("text")
    lp = LightPipeline(nlpPipeline.fit(empty_ds))
    return spark, nlpPipeline, lp


# %%
def transformData(tweets_df, spark, nlpPipeline):
    empty_df = spark.createDataFrame([['']]).toDF("text")

    pipelineModel = nlpPipeline.fit(empty_df)

    df = spark.createDataFrame(pd.DataFrame({"text":tweets_df['Tweet']}))
    result = pipelineModel.transform(df)
    
    return result

# %% [markdown]
# ## Print and Writing Results

# %%
def printTransformedResultsUsingSpark(result):
    return result.select(F.explode(F.arrays_zip('document.result', 'sentiment.result')).alias("cols")) \
    .select(F.expr("cols['0']").alias("document"),
            F.expr("cols['1']").alias("sentiment")).show(truncate=False)

# %%
def concateResults(result, tweets_df):
    pd_df = result.select(F.explode(F.arrays_zip('document.result', 'sentiment.result')).alias("cols")) \
    .select(F.expr("cols['0']").alias("document"),
            F.expr("cols['1']").alias("sentiment")).toPandas()

    pred_sentis= list(pd_df['sentiment'])
    resultDF = tweets_df.copy()
    resultDF.insert(0, 'SparkNLP_Preds', pred_sentis)
    # tweets_df.head()
    return resultDF
    # tweets_df.to_csv('tweets_with_predicted_sentiment.csv', index=False)
# %% 
# @title driver function
def doEverything(rawData = readAndShowData()):
    print(rawData.head())
    spark, nlpPipeline, lightPipeline = startSparkAndPreparePipeline()
    # transform data
    result = transformData(rawData, spark, nlpPipeline)

    # transform results of Spark to Pandas DataFrame
    transformedData = concateResults(result, rawData)
    return transformedData