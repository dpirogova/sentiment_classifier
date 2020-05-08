import csv
import random
import nltk

nr_of_folds = 10 #  the data set is split into 10 folds, which gives a 90:10 train/test split in each fold

# number and types of features to use for every language
# it has been made to set the amount of features for each language individually
use_words =                    {'en': True,  'de': True,  'fr': True,  'pt': True}
common_words_threshold =       {'en': 1000,  'de': 1000,  'fr': 1200,  'pt': 1200}
use_bigrams =                  {'en': False, 'de': True,  'fr': False, 'pt': True}
common_bigrams_threshold =     {'en': 1000,  'de': 3000,  'fr': 500,   'pt': 3000}
use_3grams =                   {'en': True,  'de': False, 'fr': False, 'pt': True}
common_char_3grams_threshold = {'en': 2700,  'de': 3000,  'fr': 5000,  'pt': 1000}
use_4grams =                   {'en': False, 'de': True,  'fr': False, 'pt': False}
common_char_4grams_threshold = {'en': 3500,  'de': 3000,  'fr': 1400,  'pt': 3000}
use_5grams =                   {'en': True,  'de': True,  'fr': True,  'pt': True}
common_char_5grams_threshold = {'en': 1900,  'de': 3000,  'fr': 1300,  'pt': 1300}

# tweet preprocessing based on GloVe, https://gist.github.com/tokestermw/cb87a97113da12acb388
import re
FLAGS = re.MULTILINE | re.DOTALL

def tweet_preprocessing(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`-]?"
    # function to make the code less repetitive
    
    def hashtag(text):
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            result = " {} ".format(hashtag_body.lower())
        else:
            result = " ".join([""] + [re.sub(r"([A-Z])",r" \1", hashtag_body, flags=FLAGS)])
        return result

    def allcaps(text):
        text = text.group()
        return text.lower() + " "
    
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"/"," / ")
    text = re_sub(r"<3","<heart>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    # text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower() #tweet pre-processing helps us to increase the accuracy

# converting a (preprocessed) tweet to a featureset
def tweet_features(tweet):
    features = {}
    
    #use_words
    words_list = set(nltk.word_tokenize(tweet))
    for word in top_words:
        features['contains({})'.format(word)] = (word in words_list)
            
    #use_bigrams
    bigrams_list = set(str(bigram) for bigram in nltk.bigrams(words_list))
    for bigram in top_bigrams:
        features['contains({})'.format(bigram)] = (bigram in bigrams_list)
            
    #use_char_4grams:
    review_char_3grams = [str(ngram) for ngram in nltk.ngrams(tweet, 3, pad_left=True, left_pad_symbol=' ')]
    for ngram in top_char_3grams:
        features['contains({})'.format(ngram)] = (ngram in review_char_3grams)
    
    #use_char_4grams:
    review_char_4grams = [str(ngram) for ngram in nltk.ngrams(tweet, 4, pad_left=True, left_pad_symbol=' ')]
    for ngram in top_char_4grams:
        features['contains({})'.format(ngram)] = (ngram in review_char_4grams)

    #use_char_5grams:
    review_char_5grams = [str(ngram) for ngram in nltk.ngrams(tweet, 5, pad_left=True, left_pad_symbol=' ')]
    for ngram in top_char_5grams:
        features['contains({})'.format(ngram)] = (ngram in review_char_5grams)
        
    return features

# loading a file of classified tweets, doing some preprocessing
def load_sentiment(filename):
    pairs = []
    with open(filename, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=";")
        next(reader) # first row, contains a header 
        for row in reader:
            pairs.append( (tweet_preprocessing(row[2]), row[0]) ) # (tweet text, sentiment) pair
    return pairs

# classifying a file for submission in the CodaLab competition format 
def classify_file(classifier, in_filename, out_filename):
    with open(in_filename, "r", encoding="utf-8") as in_file:
        with open(out_filename, "w", encoding="utf-8") as out_file:
            reader = csv.reader(in_file, delimiter=";")
            next(reader) # skipping header
            for row in reader:
                tweet_id = row[0]
                tweet = tweet_preprocessing(row[1])
                out_file.write(tweet_id + 
                        ("_pos" if classifier.classify(tweet_features(tweet)) == "positive" else "_neg") + "\r\n")

# convert a tweet list to a list of featuresets
def tweets_to_featuresets(tweets):
    featuresets = []
    for (tweet, tag) in tweets:
        featuresets.append( (tweet_features(tweet), tag) )
    return featuresets

# split the data set into train and development folds
def get_train_and_test(data_set, fold, nr_of_folds):
    test_begin = int(fold * len(data_set) / nr_of_folds)
    test_end = int((fold + 1) * len(data_set) / nr_of_folds)
    test_set = data_set[test_begin : test_end]
    train_set = data_set[:test_begin] + data_set[test_end:]
    return (train_set, test_set)
            
for lang in ["en", "de", "fr", "pt"]: #looking through languages
    data_set = load_sentiment(lang + "_sentiment_train.csv")
    
    random.seed(42) # we need to shuffle texts; the random seed chosen arbitrarily 
    random.shuffle(data_set)
    nr_tweets = len(data_set)
    
    print("Language: " + lang)
    print("Total " + str(nr_tweets) + " tweets")

    # preparing lists for the further feature extraction
    if use_words[lang]:
        fd_words = nltk.FreqDist(word for (tweet, category) in data_set for word in nltk.word_tokenize(tweet))
        top_words = [word for (word, freq) in fd_words.most_common(common_words_threshold[lang])]
    else:
        top_words = []
    
    if use_bigrams[lang]:
        fd_bigrams = nltk.FreqDist(str(bigram) for (tweet, category) in data_set for bigram in nltk.bigrams(nltk.word_tokenize(tweet)))
        top_bigrams = [bigram for (bigram, freq) in fd_bigrams.most_common(common_bigrams_threshold[lang])]
    else:
        top_bigrams = []

    if use_3grams[lang]:
        fd_char_3grams = nltk.FreqDist( str(ngram)
              for (tweet, category) in data_set
                    for ngram in nltk.ngrams(tweet.lower(), 3, pad_left=True, left_pad_symbol=' '))
        top_char_3grams = [ngram for (ngram, freq) in fd_char_3grams.most_common(common_char_3grams_threshold[lang])]
    else:
        top_char_3grams = []

    if use_4grams[lang]:
        fd_char_4grams = nltk.FreqDist( str(ngram)
              for (tweet, category) in data_set
                    for ngram in nltk.ngrams(tweet.lower(), 4, pad_left=True, left_pad_symbol=' '))
        top_char_4grams = [ngram for (ngram, freq) in fd_char_4grams.most_common(common_char_4grams_threshold[lang])]
    else:
        top_char_4grams = []
    
    if use_5grams[lang]:
        fd_char_5grams = nltk.FreqDist( str(ngram)
              for (tweet, category) in data_set
                    for ngram in nltk.ngrams(tweet.lower(), 5, pad_left=True, left_pad_symbol=' '))
        top_char_5grams = [ngram for (ngram, freq) in fd_char_5grams.most_common(common_char_5grams_threshold[lang])]
    else:
        top_char_5grams = []
    
    featuresets = tweets_to_featuresets(data_set)
    
    avg_accuracy = 0 #we are using the cross-validation method to check the accuracy
    for fold in range(nr_of_folds): 
        train_featuresets, development_featuresets = get_train_and_test(featuresets, fold, nr_of_folds)
                
        classifier = nltk.NaiveBayesClassifier.train(train_featuresets)
        
        #print(nltk.classify.accuracy(classifier, train_featuresets))
        accuracy = nltk.classify.accuracy(classifier, development_featuresets)
        print("fold nr " + str(fold) + ", accuracy " + str(accuracy))
        avg_accuracy += accuracy
    avg_accuracy /= nr_of_folds
    print("cross-validated accuracy " + str(avg_accuracy)) 
    
    # we may train the classifier on a full data set now
    classifier = nltk.NaiveBayesClassifier.train(featuresets)

    classify_file(classifier, lang + "_sentiment_test.csv", lang + "_test.txt")
    
    classifier.show_most_informative_features(200) 
