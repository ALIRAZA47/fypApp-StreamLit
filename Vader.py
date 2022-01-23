from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid_obj = SentimentIntensityAnalyzer()
def computeSentiment(rawText):
    # Create a SentimentIntensityAnalyzer object.
    
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(rawText)
    if sentiment_dict['compound'] > 0:
        sentiment = "**Sentiment::** Positive :smiley: "
    elif sentiment_dict['compound'] < 0:
        sentiment = "**Sentiment::** Negative :angry: "
    else:  # polarity == 0
        sentiment = "**Sentiment::** Neutral 😐 "
    
    return sentiment, sentiment_dict


labels = ['negative', 'neutral', 'positive']
def labelSentiment(sentiment):
    if sentiment < 0:
        return labels[0]
    elif sentiment == 0:
        return labels[1]
    else:
        return labels[2]
def analyzeBatch(tweets_df):    
    tweets_df['Vader_Preds'] = tweets_df['Tweet'].apply(lambda tweet: labelSentiment(sid_obj.polarity_scores(tweet)['compound']))
    cols = list(tweets_df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = tweets_df[cols]
    return df