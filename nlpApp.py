import streamlit as st
from textblob import TextBlob
import pandas as pd
import altair as alt #For Visualization
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn
import os
import openAI



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

def sentimentAnalysisAfinn(rawText):
    #instantiate afinn
    afn = Afinn()
    score = afn.score(rawText) 
    return score 

def covent_to_dataFrameVADER(sentiDic):
    sentiDataFrame = pd.DataFrame(sentiDic.items(), columns=['metric', 'value'])
    return sentiDataFrame

def covent_to_dataFrame(sentiment):
    #print(sentiDic)
    #print(type(sentiDic)) #<class 'textblob.en.sentiments.Sentiment'>
    sentiDic = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiDataFrame = pd.DataFrame(sentiDic.items(), columns=['metric', 'value'])
    return sentiDataFrame

def sentiAnalysisVADER(rawText):
    	# Create a SentimentIntensityAnalyzer object.
	sid_obj = SentimentIntensityAnalyzer()

	# polarity_scores method of SentimentIntensityAnalyzer
	# object gives a sentiment dictionary.
	# which contains pos, neg, neu, and compound scores.
	sentiment_dict = sid_obj.polarity_scores(rawText)
	
	#print("Overall sentiment dictionary is : ", sentiment_dict)
	#print("sentence was rated as ", sentiment_dict['neg']*100, "% Negative")
	#print("sentence was rated as ", sentiment_dict['neu']*100, "% Neutral")
	#print("sentence was rated as ", sentiment_dict['pos']*100, "% Positive")
	#print("Sentence Overall Rated As", end = " ")
    
	return sentiment_dict


  
def main():
    customizeUI()
    #st.image('./cool.png')
    st.title("Deep Research App")
    st.markdown("**Welcome** 😊")
    
    menu = ["Home", "ML", "Comparison", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Rule-Based Sentiment Analysis")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here:")
            submit_button = st.form_submit_button(label='Analyze')
            
        # layout
        col1,col2,col3 = st.columns(3) 
        if submit_button:
            with col1:
                # st.info("TextBlob Sentiment Analysis (sentiwordnet)") #TextBlob Sentiment Analysis (sentiwordnet)
                # sentiment = TextBlob(raw_text).sentiment
                # st.write(sentiment)
                # #Emoji
                # if sentiment.polarity > 0:
                #     st.markdown("**Sentiment::** Positive :smiley: ")
                # elif sentiment.polarity < 0:
                #     st.markdown("**Sentiment::** Negative :angry: ")
                # else: #polarity == 0   
                #     st.markdown("**Sentiment::** Neutral 😐 ")

                # #DataFrame TextBlob Output
                # resultDataFrame = covent_to_dataFrame(sentiment)
                # st.dataframe(resultDataFrame)
                # # Visualization
                # c = alt.Chart(resultDataFrame).mark_bar().encode(
				# 	x='metric',
				# 	y='value',
				# 	color='metric')
                # st.altair_chart(c,use_container_width=True)
                
                #***************************************************************
                
                # #Openai GPT3
                st.info("Openai GPT3") #Openai GPT-3 <--------------------------------------
                sentimentOpen,rawResponse = openAI.sentimentAnalysis(raw_text)
                st.write(rawResponse)
                #Emoji
                if sentimentOpen == "Positive":
                    st.markdown("**Sentiment::** Positive :smiley: ")
                elif sentimentOpen == "Negative":
                    st.markdown("**Sentiment::** Negative :angry: ")
                else: #polarity == 0   
                    st.markdown("**Sentiment::** Neutral 😐 ")
                #*************************************************************** 
            
                
            with col2:
                st.info("VADER Sentiment Analysis") #VADER Sentiment Analysis
                sentimentDict = sentiAnalysisVADER(raw_text)
                st.write(sentimentDict)
                # decide sentiment as positive, negative and neutral
                if sentimentDict['compound'] >= 0.05 :
                    st.markdown("**Sentiment::** Positive :smiley: ")

                elif sentimentDict['compound'] <= - 0.05 :
                    st.markdown("**Sentiment::** Negative :angry: ")

                else :#polarity < +0.05 and > -0.05
                    st.markdown("**Sentiment::** Neutral 😐 ")

                #DataFrame TextBlob Output
                resultDataFrame = covent_to_dataFrameVADER(sentimentDict)
                st.dataframe(resultDataFrame)
                # Visualization
                c = alt.Chart(resultDataFrame).mark_bar().encode(
					x='metric',
					y='value',
					color='metric')
                st.altair_chart(c,use_container_width=True)
            
            with col3:
                st.info("Afinn Sentiment Analysis") #TextBlob Sentiment Analysis (sentiwordnet)
                sentiment = sentimentAnalysisAfinn(raw_text)
                st.write(sentiment)
                #Emoji
                if sentiment > 0:
                    st.markdown("**Sentiment::** Positive :smiley: ")
                elif sentiment < 0:
                    st.markdown("**Sentiment::** Negative :angry: ")
                else: #polarity == 0   
                    st.markdown("**Sentiment::** Neutral 😐 ")

                

    elif choice == "ML": 
        st.subheader("ML-Based Sentiment Analysis")
               
    elif choice == "Comparision": 
        st.subheader("Comparision")
    else:
        st.subheader("About")
        
if __name__ == "__main__":
    main()