# -*- coding: utf-8 -*-
import nltk
import re
from sklearn.naive_bayes import MultinomialNB
import numpy as np
#Clean the data: preprocessing
#replace @user with ||U||,  re.search(r'^@\w+', "@user")
#replace #tag with ||T||, re.search(r'^#\w+', "#tag")
#replace URL with URL. WARNING: URL got punctations. re.search(r'^(http)[\w.-/:]+', "https:/")
#If the word ends with a punctations (? . ! , :),remove them all (more than one po#pitchda…ssible) re.search(r'.+[\.\?,:;!]+$', "hello??")
#If last character is special character of twitter, remove the word.
def preprocessing_raw_tweets(raw_tweet):
    tweet = []
    for word in raw_tweet:
        if(re.search(r".+\xa6$", word)):
            continue
        elif(re.search(r'^@\w+', word)):
            continue
            #tweet.append(re.sub(r'^@[\w\.,;:\?]+', "|U|", word))
        elif(re.search(r'^#\w+', word)):
            continue
            #tweet.append(re.sub(r'^#[\w\.,;:\?]+', "|T|", word))
        elif(re.search(r'^(http)[\w.-/:]+', word)):
            continue
            #tweet.append(re.sub(r'^(http)[\w.-/:;,\?]+', "URL", word))
        elif(re.search(r'.+[\.\?,:;!]+$', word)):
            tweet.append(re.sub(r'[\.\?,:;!]+$', "", word))
        else:
          tweet.append(word)
    
    return tweet


#appending all the tweet words       
def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words

#getting the frequency distribution of words in the tweet
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    return wordlist

#extracting feature value for a tweet
def extract_features_values(tweet):
    feature_values = []
    for word in word_features:
        feature_values.append(tweet.count(word))
    return feature_values


with open('/home/piyushb/Documents/Sentimental analysis/Test 1/labeledDataWithCleaning.csv','r') as tsv:
    raw_data = [line.strip().split('\t') for line in tsv]

data = []

for (sentiment, words) in raw_data:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    data.append((words_filtered, sentiment))

data_processed = []
raw_tweets = []
polarity = []

for (words, sentiment) in data:
    words_preprocessed = preprocessing_raw_tweets(words)
    data_processed.append((words_preprocessed, sentiment))
    raw_tweets.append(words_preprocessed)
    polarity.append(sentiment)

#print data_processed[:10]
#print raw_tweets[:10]
#print polarity[:10]

all_words = get_words_in_tweets(data_processed)
words_frequency = get_word_features(all_words)

word_features = []
for word in set(all_words):
    if words_frequency[word] >= 2:
        word_features.append(word)

X_train_temp = []
for (tweet, sentiment) in data_processed:
    X_train_temp.append(extract_features_values(tweet))

X_train = np.array([np.array(xi) for xi in X_train_temp])
y_train = np.array(polarity)

print type(X_train)
print type(y_train)

clf = MultinomialNB()
clf.fit(X_train, y_train)