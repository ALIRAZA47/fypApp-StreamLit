import pandas as pd
import openai
import streamlit as st

def sentimentAnalysis(rawText):    
    openai.api_key = "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"
    trainDatePlusRawText = "This is a Sentence sentiment classifier\n\n\nSentence: \"I loved the new Batman movie!\"\nSentiment: Positive\n###\nSentence: \"I hate it when my phone battery dies.\"\nSentiment: Negative\n###\nSentence: \"My day has been 👍\"\nSentiment: Positive\n###\nSentence: \"This is the link to the article\"\nSentiment: Neutral\n###\nSentence: \"This new music video blew my mind\"\nSentiment: Positive\n###\nSentence: \""
    trainDatePlusRawText = trainDatePlusRawText + rawText + "\"\nSentiment:"
    response = openai.Completion.create(
    engine="davinci",prompt=trainDatePlusRawText,temperature=0.3,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0,
    stop=["###"])
    return response["choices"][0]["text"][:-1].strip(), response
    
def showResults(rawText):
    sentimentOpen,rawResponse = sentimentAnalysis(rawText)
    st.write(rawResponse)
    #Emoji
    if sentimentOpen == "Positive":
        st.markdown("**Sentiment::** Positive :smiley: ")
    elif sentimentOpen == "Negative":
        st.markdown("**Sentiment::** Negative :angry: ")
    else: #polarity == 0   
        st.markdown("**Sentiment::** Neutral 😐 ")


