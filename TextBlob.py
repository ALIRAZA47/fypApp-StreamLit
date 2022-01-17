from textblob import TextBlob
import streamlit as st
import pandas as pd
import altair as alt #For Visualization

def computerSentiment(text):
    fullRes = TextBlob(text)
    sentimentRes = fullRes.sentiment
    sentiment = ""
    st.write(sentimentRes)
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

    sentiment, fullResult = computerSentiment(inputText)
    st.markdown(sentiment)

    # DataFrame TextBlob Output
    # sentiDic = {'polarity': fullResult.polarity, 'subjectivity': fullResult.subjectivity}
    # sentiDataFrame = pd.DataFrame(sentiDic.items(), columns=['metric', 'value'])
    
    resultDataFrame = pd.DataFrame({'polarity': fullResult.polarity,'subjectivity': fullResult.subjectivity}.items(), columns=['metric', 'value'])
    st.dataframe(resultDataFrame)
    # Visualization
    # c = alt.Chart(resultDataFrame).mark_bar().encode(
    # x='metric',
    # y='value',
    # color='metric')
    # st.altair_chart(c, use_container_width=True)
