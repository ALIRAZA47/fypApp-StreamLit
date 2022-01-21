import tweepy
import ssl
import pandas as pd

# ssl._create_default_https_context = ssl._create_unverified_context

# Set API Key and Access tokens
consumer_key = "Xb15yG3eW4NWtco9ZXGeDff2z"
consumer_secret = "eMpSXheAWshWWLJQfTTukEwzMUELTLKOsTaccU2gVk7ZNgW9wE"
access_token = "821276337239363584-8bMyxuw9QEKpyc5e5Omq7ahTOb1N60t"
access_token_secret = "cswpR9KeYl51XfFF2xUSNnuAMfEBJjgfRHxgoY7KfhN8C"
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