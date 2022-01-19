from asyncore import write
from typing import Text
from xml.dom import ValidationErr
import streamlit as st
from st_aggrid import AgGrid

import pandas as pd
import altair as alt #For Visualization
import os
import validators
import TweetsScrapper  #For Scrapping Tweets
import TextBlob
import OpenAI
# spark nlp imports
import TextPreprocessing
import SparkNLP  # our API
from sparknlp.base import LightPipeline, Pipeline
import sparknlp


# Global Variables Here
sentimentDict = {'positive':"**Sentiment::** Positive :smiley: ",
                'negative':"**Sentiment::** Negative :angry: ",
                'neutral':"**Sentiment::** Neutral 😐 "}


# CustomizingUI
def customizeUI():

    #Set the page title and icon
    st.set_page_config(
        page_title="Deep NLP",
        page_icon="🔥", 
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
    )

    #For Hiding the menu button
    st.markdown(""" <style> 
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)

    #For Condensing the layout
    padding = 1
    st.markdown(f""" <style>
        .reportview-container .main .block-container{{
            padding-top: {padding}rem;
            padding-right: {padding}rem;
            padding-left: {padding}rem;
            padding-bottom: {padding}rem;
        }} </style> """, unsafe_allow_html=True)

    st.image('./nlp2.jpeg') # Img Reference: https://editor.analyticsvidhya.com/uploads/49583NLP-scaled-1-2048x771.jpeg

# Main Function
  
def main():
    customizeUI()
    #st.image('./cool.png')
    st.title("Deep Research App")
    st.markdown("**Welcome** 😊")
    
    menu = ["Comparative Analysis", "Spark NLP", "BERT", "Comparison", "Live Twitter Feed"]
    selectedTab = st.sidebar.selectbox("Menu", menu)
    
    if selectedTab == "Comparative Analysis":
        st.subheader("Rule-Based Sentiment Analysis")
        with st.form(key='nlpForm'):
            inputText = st.text_area("Enter Text Here:")
            submit_button = st.form_submit_button(label='Analyze')
            
        # layout
        col1,col2,col3 = st.columns(3) 
        if submit_button:
            spark, fullPipline, lightPipeline = SparkNLP.startSparkAndPreparePipeline()    #Spark NLP intialization
            with col1:
                # Open AI Sentiment Analysis
                st.info("Openai GPT3") #Openai GPT-3 <--------------------------------------
                OpenAI.showResults(inputText) 
                
            with col2:
                st.info("TextBlob Sentiment Analysis (sentiwordnet)")
                TextBlob.showResults(inputText)
            
            with col3:
                st.info("SparkNLP's Pretrained Pipeline (sentimentdl_user_twitter )") #Spark NLP Pipeline<--------------------------------------
                result = lightPipeline.annotate(inputText)
                st.write(result)
                st.markdown(sentimentDict[result['sentiment'][0]])                

    elif selectedTab == "Spark NLP": 
        st.subheader("John Snow Labs' Spark NLP Sentiment Analysis")
        rawData = SparkNLP.readAndShowData()
        # st.header("Raw Comments (Twitter)", 
        st.markdown('<h4 style="text-align: center;:"> \
                    Raw Comments (Twitter) \
                    </h4>',
                    unsafe_allow_html=True)
        AgGrid(rawData)
        
        # Analyze comments
        sparkAnalyzeBtn = st.button("Analyze Comments")
        if sparkAnalyzeBtn: #if the button is clicked
            st.markdown('<h4 style="text-align: center;:"> \
                    Resultant Data (with Predicted Sentiments) \
                    </h4>',
                    unsafe_allow_html=True)
            resultDF, confMatrix = SparkNLP.doEverything()
            AgGrid(resultDF)
            st.write(confMatrix)
    
    elif selectedTab == "Comparision": 
        st.subheader("About")
    
    elif selectedTab == "BERT": 
        st.subheader("BERT Analysis")
    
    elif selectedTab == "Live Twitter Feed": 
        st.subheader("Live Twitter Feed")
        # Form
        with st.form(key='twitterLinkForm'):
            inputTweetLink = st.text_area("Enter Tweet Link Here:")
            fetchTweetsBtn = st.form_submit_button(label='Fetch Tweets')
        
        if fetchTweetsBtn:
            if validators.url(inputTweetLink):
                # URL is valid
                st.info("Fetching Tweets...")
                tweets = pd.read_csv('replies_clean.csv')
                # tweets = TweetsScrapper.fetchTweets(inputTweetLink)
                st.subheader("(Raw) Replies on {} by {} " .format(inputTweetLink.split('/')[3], inputTweetLink.split('/')[5]))
                AgGrid(tweets)
                
                # TODO: Add Live Sentiment Analysis here
                
                
            else:
                st.error("Please Enter a Valid URL")
  
if __name__ == "__main__":
    main()