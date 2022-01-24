from asyncore import write
from builtins import breakpoint

from cgitb import enable
from typing import Text
from xml.dom import ValidationErr
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder


import HelperFuncs
import pandas as pd
import altair as alt #For Visualization
import os
import validators
import OpenAI
import Vader

import TweetsScrapper  #For Scrapping Tweets
import TextBlob
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
        page_title="ICA",
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
    st.title("Intelligent Content Analyzer")
    st.markdown("**Welcome** 😊")
    
    menu = ["Comparative Analysis", "Batch Analysis", "Live Twitter Feed"]
    selectedTab = st.sidebar.selectbox("Menu", menu)
    
    if selectedTab == "Comparative Analysis":
        st.subheader("Rule-Based Sentiment Analysis")
        with st.form(key='nlpForm'):
            inputText = st.text_area("Enter Text Here:")
            submit_button = st.form_submit_button(label='Analyze')
            
        # layout
        col1,col2,col3, col4 = st.columns(4) # 4 columns
        if submit_button:
            spark, fullPipline, lightPipeline = SparkNLP.startSparkAndPreparePipeline()    #Spark NLP intialization
            with col1:
                # Open AI Sentiment Analysis
                st.info("Openai GPT3 (davinci)") #Openai GPT-3 <--------------------------------------
                OpenAI.showResults(inputText) 
                
            with col2:
                st.info("TextBlob Sentiment Analysis (sentiwordnet)")
                TextBlob.showResults(inputText)
                
            with col3:
                st.info("Vader Sentiment Analysis")
                sentimentVader, fullResVader = Vader.computeSentiment(inputText)
                st.write(sentimentVader)
            
            with col4:
                st.info("SparkNLP's (sentimentdl_user_twitter )") #Spark NLP Pipeline<--------------------------------------
                result = lightPipeline.annotate(inputText)
                # st.write(result)           #full response from spark
                st.markdown(sentimentDict[result['sentiment'][0]])                

    elif selectedTab == "Batch Analysis": 
        analyzeBtn = False
        sparkNlpCB = st.sidebar.checkbox("SparkNLP")
        textBlobCB = st.sidebar.checkbox("TextBlob")
        vaderCB = st.sidebar.checkbox("Vader")
        if sparkNlpCB or textBlobCB or vaderCB:
            # sparkNlpCB = st.sidebar.checkbox("SparkNLP")
            rawData = pd.DataFrame()
            st.subheader("Batch Analysis")
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>', unsafe_allow_html=True)
            selectedRadioBtn = st.radio("", ("Analyze Default Batch", "Analyze Custom Batch"))
            
            if selectedRadioBtn == "Analyze Default Batch":
                rawData = SparkNLP.readAndShowData()
                
            # analyze custom batch (file)
            else:
                rawData = pd.DataFrame()
                st.subheader("Analyze Custom Batch")
                #st.write("Upload a file")
                file = st.file_uploader("Upload a CSV file", type=["csv"], help="File header should be 'True Sentiment', Tweet ID and 'Tweet' respectively")
                if file is not None:
                    rawData = pd.read_csv(file, on_bad_lines='skip')
            
            # Show file preview
            if rawData.empty:
                st.warning("No Data Found")
            else:
                st.markdown('<h4 style="text-align: center;:"> \
                                Raw Comments (Twitter) \
                                </h4>',
                                unsafe_allow_html=True)
                gridOptions = HelperFuncs.buildGridOptionAgGrid(rawData)
                AgGrid(rawData, gridOptions=gridOptions, enable_enterprise_modules=True)
                # Analyze comments
                analyzeBtn = st.button("Analyze Comments")
                
            if analyzeBtn: #if the button is clicked
                resultsLoaded = False
                accSystems = []
                accList = []
                precList = []
                recallList = []
                selected_cols = rawData[['True Sentiment', 'Tweet']]
                resultDF = selected_cols.copy()
                if sparkNlpCB:
                    resultDF = SparkNLP.doEverything(resultDF)
                    sparkAcc, sparkPR, sparkRecall = HelperFuncs.computeAPR(resultDF, 'SparkNLP_Preds')
                    recallList.append(sparkRecall)
                    precList.append(sparkPR)
                    accList.append(sparkAcc)
                    accSystems.append('SparkNLP')
                    
                if textBlobCB :
                    resultDF = TextBlob.analyzeBatch(resultDF)
                    textBlobAcc, tbPR, tbRec = HelperFuncs.computeAPR(resultDF, 'TextBlob_Preds')
                    precList.append(tbPR)
                    recallList.append(tbRec)
                    accList.append(textBlobAcc)
                    accSystems.append('TextBlob')
                    
                if vaderCB:
                    resultDF = Vader.analyzeBatch(resultDF)
                    vadAcc, vaderPR, vaderRC = HelperFuncs.computeAPR(resultDF, "Vader_Preds")
                    accList.append(vadAcc)
                    precList.append(vaderPR)
                    recallList.append(vaderRC)
                    accSystems.append('Vader')
                    
                if vaderCB == False and sparkNlpCB == False and textBlobCB == False:
                    st.warning("Please select atleast one option from the checkboxes")   
                
                resultsLoaded = True
                
                if resultDF.empty:
                    st.warning("No Data Found")
                else:
                    st.markdown('<h4 style="text-align: center;:"> \
                            Resultant Data (with Predicted Sentiments) \
                            </h4>',
                            unsafe_allow_html=True)
                    gridOptions = HelperFuncs.buildGridOptionAgGrid(resultDF)
                    AgGrid(resultDF, gridOptions=gridOptions, enable_enterprise_modules=True)
                
                    # CSS to inject contained in a string
                    hide_dataframe_row_index = """
                                <style>
                                .row_heading.level0 {display:none}
                                .blank {display:none}
                                </style>
                                """

                    # Inject CSS with Markdown
                    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
                    
                    accuracyDF = pd.DataFrame({"System": accSystems, "Accuracy": accList, "Precision": precList, "Recall": recallList})
                    st.sidebar.dataframe(accuracyDF, width=900)
                    
                    
        else:
            st.error("Please select atleast one option from the checkboxes")
    
    elif selectedTab == "Live Twitter Feed": 
        st.subheader("Live Twitter Feed")
        # Form
        with st.form(key='twitterLinkForm'):
            inputTweetLink = st.text_area("Enter Tweet Link Here:")
            fetchTweetsBtn = st.form_submit_button(label='Fetch and Analyze Replies')
        analyzeBtn = False
        sparkNlpCB = st.sidebar.checkbox("SparkNLP")
        textBlobCB = st.sidebar.checkbox("TextBlob")
        vaderCB = st.sidebar.checkbox("Vader")
        fetchedTweets = pd.DataFrame()
        
        if sparkNlpCB or textBlobCB or vaderCB:
            # sparkNlpCB = st.sidebar.checkbox("SparkNLP")
            if fetchTweetsBtn:
                if validators.url(inputTweetLink):
                    # URL is valid
                    st.info("Fetching Tweets :hourglass:")
                    # fetchedTweets = pd.read_csv('replies_clean.csv')
                    fetchedTweets = TweetsScrapper.fetchTweets(inputTweetLink)
                    st.markdown('<h4 style="text-align: center;:"> \
                                    (Raw) Replies on tweet '+inputTweetLink.split('/')[5]+\
                                    ' by '+inputTweetLink.split('/')[3]+' \
                                </h4>',
                                    unsafe_allow_html=True)
                    gridOptions = HelperFuncs.buildGridOptionAgGrid(fetchedTweets)
                    AgGrid(fetchedTweets, gridOptions=gridOptions, enable_enterprise_modules=True)
                    
                    # TODO: Add Live Sentiment Analysis here
                else:
                    st.error("Please Enter a Valid URL")
                rawData = fetchedTweets
                # Show file preview
                if rawData.empty:
                    st.warning("No Data Found")
                    
                resultsLoaded = False
                resultDF = fetchedTweets
                
                if sparkNlpCB:
                    resultDF = SparkNLP.doEverything(resultDF)
                if textBlobCB :
                    resultDF = TextBlob.analyzeBatch(resultDF)
                if vaderCB:
                    resultDF = Vader.analyzeBatch(resultDF)
                if vaderCB == False and sparkNlpCB == False and textBlobCB == False:
                    st.warning("Please select atleast one option from the checkboxes")   
                
                resultsLoaded = True
                
                if resultDF.empty:
                    st.warning("No Data Found")
                else:
                    st.markdown('<h4 style="text-align: center;:"> \
                            Resultant Data (with Predicted Sentiments) \
                            </h4>',
                            unsafe_allow_html=True)
                    gridOptions = HelperFuncs.buildGridOptionAgGrid(resultDF)
                    AgGrid(resultDF, gridOptions=gridOptions, enable_enterprise_modules=True)
                
                    # CSS to inject contained in a string
                    hide_dataframe_row_index = """
                                <style>
                                .row_heading.level0 {display:none}
                                .blank {display:none}
                                </style>
                                """

                    # Inject CSS with Markdown
                    st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)
                    
                    
        else:
            st.error("Please select atleast one option from the checkboxes")
if __name__ == "__main__":
    main()