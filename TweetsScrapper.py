import tweepy
import ssl
import pandas as pd
import csv
import os

# ssl._create_default_https_context = ssl._create_unverified_context

# Set API Key and Access tokens
consumer_key = os.environ['TWITTER_CONSUMER_KEY']
consumer_secret = os.environ['TWITTER_CONSUMER_SECRET']
access_token =  os.environ['TWITTER_ACCESS_TOKEN']
access_token_secret = os.environ['TWITTER_ACCESS_TOKEN_SECRET']
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)


def sliceURL(text):
    url = text.split("/")
    return url[3], url[5]
# Code to scrap tweets goes here
# update these for whatever tweet you want to process replies to

def fetchTweets(inputURL):
    
    userName, tweetID = sliceURL(inputURL)
    print(tweetID, userName)
    replies=[]



    tweets = tweepy.Cursor(api.search_tweets,q='to:'+userName, result_type='recent', lang='en').items(100)

    for tweet in tweets:
        if hasattr(tweet, 'in_reply_to_status_id_str'):
            if (tweet.in_reply_to_status_id_str==tweetID):
                replies.append(tweet)
    # print(replies)
    print("-------------------1")
    tweetsDF = pd.DataFrame(data=[tweet.text for tweet in replies], columns=['Tweet'])
    tweetsDF.to_csv('replies_clean.csv', index=False)
    return tweetsDF