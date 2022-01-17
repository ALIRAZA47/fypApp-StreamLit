from typing import Text
import streamlit as st
import pandas as pd
import altair as alt #For Visualization
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn
import os
import OpenAI
import TextBlob
from st_aggrid import AgGrid
# from sparknlp.base import LighPipiLine




#CustomizingUI
def customizeUI():
    #<-------------------------------------------------------------------------->
    # customising the Streamlit UI start <-----------------------------------------
    # https://towardsdatascience.com/5-ways-to-customise-your-streamlit-ui-e914e458a17c 

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

    
    # customising the Streamlit UI end <-----------------------------------------
    #<-------------------------------------------------------------------------->
    #<-------------------------------------------------------------------------->


#Textblob:
#By default, it calculates average polarity and subjectivity
#over each word in a given text using a dictionary of adjectives
#and their hand-tagged scores. It actually uses pattern library
#for that, which takes the individual word scores from sentiwordnet.

# *******************************************************************

# *******************************************************************
  
def main():
    customizeUI()
    #st.image('./cool.png')
    st.title("Deep Research App")
    st.markdown("**Welcome** 😊")
    
    menu = ["Home", "Spark NLP", "BERT", "Comparison", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Rule-Based Sentiment Analysis")
        with st.form(key='nlpForm'):
            inputText = st.text_area("Enter Text Here:")
            submit_button = st.form_submit_button(label='Analyze')
            
        # layout
        col1,col2,col3 = st.columns(3) 
        if submit_button:
            with col1:
                # Open AI Sentiment Analysis
                OpenAI.showResults(inputText) 
                
            with col2:
                TextBlob.showResults(inputText)
            
            with col3:
                pass
                

    elif choice == "Spark NLP": 
        st.subheader("John Snow Labs' Spark NLP Sentiment Analysis")
        import SparkNLP
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
            resultDF = SparkNLP.doEverything()
            AgGrid(resultDF)
        
            # Visualize Results
            # st.markdown('<h4 style="text-align: center;:"> \
            #             Results '+ str(resultDF.sentiment.value_counts()) +' \
            #             </h4>',
            #         unsafe_allow_html=True)
        
        
    elif choice == "Comparision": 
        st.subheader("About")
        
if __name__ == "__main__":
    main()