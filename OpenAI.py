import pandas as pd
import os
import openai
import streamlit as st

def sentimentAnalysis(rawText):    
    openai.api_key = "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"
    trainDatePlusRawText = "This is a Sentence sentiment classifier\n\n\nSentence: \"I loved the new Batman movie!\"\nSentiment: Positive\n###\nSentence: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nSentence: \"My day has been 👍\"\nSentiment: Positive\n###\nSentence: \"This is the link to the article\"\nSentiment: Neutral\n###\nSentence: \"This new music video blew my mind\"\nSentiment: Positive\n###\nSentence: \""
    trainDatePlusRawText = trainDatePlusRawText + rawText + "\"\nSentiment:"
    response = openai.Completion.create(
    model="davinci:ft-personal-2022-01-23-11-55-31",
    prompt=trainDatePlusRawText,temperature=0.7,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0,
    stop=["###"])
    return response["choices"][0]["text"][:-1].strip()
    
def showResults(rawText):
    sentimentOpen,rawResponse = sentimentAnalysis(rawText)
    # st.write(rawResponse)  #full response from openAI API
    #Emoji
    sentimentOpen = sentimentOpen.lower()
    if sentimentOpen == "positive":
        st.markdown("**Sentiment::** Positive :smiley: ")
    elif sentimentOpen == "negative":
        st.markdown("**Sentiment::** Negative :angry: ")
    elif sentimentOpen == "neutral":    
        st.markdown("**Sentiment::** Neutral 😐 ")
    else:
        st.markdown("**Sentiment::** None 😐 ")
        
def batchSentimentAnalysis(tweets_df):
    tweets_df['GPT3_Preds'] = tweets_df['Tweet'].apply(lambda tweet: sentimentAnalysis(tweet))
    cols = list(tweets_df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = tweets_df[cols]
    return df


