# %% [markdown]
# # Text Preprocessing Using Spark NLP

# %%
import os
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.common import *
from sparknlp.base import *
import sparknlp
spark = sparknlp.start()

# %%
import pandas as pd
import re
import string

# %% [markdown]
# ## Reading Dataset

# %%
# colnames=['tweet_id', 'sentiment', 'text'] 
# tweets_df = pd.read_csv('3points_dataset_tweets.csv',sep="\t" , header=None, names=colnames) 
# tweets_df.head()

# %% [markdown]
# ## Removing 'Not Available', Links and Metions


# %%
def removeLinks(text):
    link_regex    = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links         = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')    
    return text

def cleanAtTheRateAndHashtags(text):
    junk = ['@', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    words = text.split()
    res =[]
    for word in words:
        if word[0] == '#' or word[0] == '@':
            newWord = []
            for letter in word:
                if letter not in junk:
                    newWord.append(letter)
            res.append(''.join(newWord))
        else:
            res.append(word)
    return " ".join(res)
def cleanseText(text):
    text = removeLinks(text)
    entity_prefixes = ['@', '#','\'', '-']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    txt = ' '.join(words)
    return txt

# %%
def cleanTextDataFrame(tweets_df):
    tweets_df = tweets_df[(tweets_df.text != "Not Available")]
    tweets_df['text'] = tweets_df['text'].apply(lambda x: cleanseText(x))
    tweets_df.head()


# %% [markdown]
# ## Spell Checker

# %%
def prepareSpellPipeline():
    # from sparknlp_jsl.annotator import *

    documentAssembler = DocumentAssembler()\
        .setInputCol('text')\
        .setOutputCol('document')
        
    tokenizer = RecursiveTokenizer()\
        .setInputCols(['document'])\
        .setOutputCol('token')\
        .setPrefixes(["\"", "(", "[", "\n"])\
        .setSuffixes([".", ",", "?", ")","!", "‘s"])
                    
    spellModel = ContextSpellCheckerModel\
        .pretrained()\
        .setInputCols('token')\
        .setOutputCol('checked')
        
    finisher = Finisher()\
        .setInputCols('checked')

    spellPipline = Pipeline(stages = [
    documentAssembler,
    tokenizer,
    spellModel,
    finisher
    ])

    # let's create an empty dataframe just to call fit()
    empty_ds = spark.createDataFrame([[""]]).toDF("text")
    lp = LightPipeline(spellPipline.fit(empty_ds))

    return lp, spellPipline

# %%
# add some more, in case we need them
# spellModel.updateVocabClass('_NAME_', ['ali', 'khan', 'muhammad', 'ahmad', 'zara', 'zoya', 'ayesha'], True)# Let's see what we get now
# spellModel.updateRegexClass('_DATE_', '(january|february|march|april|may|june|july|august|september|october|november|december)-[0-31]')
# sample = 'We are going to meet ali ahmad mahnoor on the october-3'
# lp.annotate(sample)

# %%
# empty_df = spark.createDataFrame([['']]).toDF("text")
# spellPipelineModel = spellPipline.fit(empty_df)

# %%
# spark_df = spark.createDataFrame(pd.DataFrame({"text": tweets_df['text']}))


# %%
# tweets_df['corrected_text'] = spark_df.withColumn("corrected_text", spellPipelineModel.transform(spark_df).select("checked").collect()[0][0])

# %%
def correctSpells(tweets_df):
    lp, spellPipeline = prepareSpellPipeline()
    tweets_df['corrected_text'] = tweets_df['text'].apply(lambda x: " ".join(lp.annotate(x)['checked']))
    return tweets_df


# %% [markdown]
# ## Normalizing Text by Removing Punctuation

# %%
string.punctuation

# %%
def preparePunctuationPipeline():
    documentAssembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

    tokenizer = Tokenizer() \
        .setInputCols(["document"]) \
        .setOutputCol("token")
        
    normalizer = Normalizer() \
        .setInputCols(["token"]) \
        .setOutputCol("normalized")\
        .setLowercase(True)\
        .setCleanupPatterns(["[^\w\d\s]"])\
        # remove punctuations (keep alphanumeric chars)
        # if we don't set CleanupPatterns, it will only keep alphabet letters ([^A-Za-z])

    nlpPipeline = Pipeline(stages=[
        documentAssembler, 
        tokenizer,
        normalizer
    ])

    empty_df = spark.createDataFrame([['']]).toDF("text")
    pipelineModel = nlpPipeline.fit(empty_df)
    empty_ds = spark.createDataFrame([[""]]).toDF("text")
    lp = LightPipeline(pipelineModel)
    
    return pipelineModel, lp

# %%
# pipelineModel.stages

# %%
def removePunctuation(tweets_df):
    fp, lp = preparePunctuationPipeline()
    tweets_df['normalized_text'] = tweets_df['corrected_text'].apply(lambda x: " ".join(lp.annotate(x)))
    return tweets_df

# %%
# result.show(5, truncate=20)

# %%
# result.select('normalized.result').take(8)

# lp, fp = prepareSpellPipeline()
# print(" ".join(lp.annotate("I'm a studentt")['checked']))

