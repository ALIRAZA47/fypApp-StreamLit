from textblob import TextBlob
import streamlit as st
import pandas as pd
import altair as alt #For Visualization

def computeSentiment(text):
    fullRes = TextBlob(text)
    sentimentRes = fullRes.sentiment
    sentiment = ""
    # st.write(sentimentRes)
    # Emoji
    if sentimentRes.polarity > 0:
        sentiment = "**Sentiment::** Positive :smiley: "
    elif sentimentRes.polarity < 0:
        sentiment = "**Sentiment::** Negative :angry: "
    else:  # polarity == 0
        sentiment = "**Sentiment::** Neutral 😐 "
        
    return sentiment, sentimentRes

# TextBlob Sentiment Analysis (sentiwordnet)
def showResults(inputText):

    sentiment, fullResult = computeSentiment(inputText)
    st.markdown(sentiment)

    # DataFrame TextBlob Output
    # sentiDic = {'polarity': fullResult.polarity, 'subjectivity': fullResult.subjectivity}
    # sentiDataFrame = pd.DataFrame(sentiDic.items(), columns=['metric', 'value'])
    
    resultDataFrame = pd.DataFrame({'polarity': fullResult.polarity,'subjectivity': fullResult.subjectivity}.items(), columns=['metric', 'value'])
    # st.dataframe(resultDataFrame)

# analyze batch of text
labels = ['negative', 'neutral', 'positive']
def labelSentiment(sentiment):
    if sentiment < 0:
        return labels[0]
    elif sentiment == 0:
        return labels[1]
    else:
        return labels[2]
def analyzeBatch(tweets_df):    
    tweets_df['TextBlob_Preds'] = tweets_df['Tweet'].apply(lambda tweet: labelSentiment(TextBlob(tweet).sentiment.polarity))
    cols = list(tweets_df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = tweets_df[cols]

    return df
