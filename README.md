# sentiment_classifier
Script for the classification taks

Data for the classification task were given by the computer science faculty of the TU Darmstadt.

## Project description 

The aim of the presented code is to solve the classification task. The chosen corpus consists of short text messages in four languages. All these messages are no longer than 140 symbols. These text messages include different spelling errors, slang words and phrases and abbreviations, which make text messages more difficult for classification. In the training part of the corpus all messages are marked with one of two categories; they can be “positive” or “negative”. The goal of the classifier is to mark other text messages that had not been seen before with these two labels.

The data pre-processing could improve results so hashtags and smileys presented in the tweets were replaced with special tokens based of the existing code (https://gist.github.com/tokestermw/cb87a97113da12acb388).

Bigrams and words were chosen as features for the classifier. Moreover, when a grammar system of a language is not simple and a language includes a lot of root words and misspellings (https://habr.com/ru/post/149605/), it can be useful to try 3-, 4- and 5-grams as features during the classifier learning, so symbol n-grams were tried in the code.

Hence, the types of the features (words, word bigrams and three types of symbol n- grams) and their amounts based of the frequency of their appearance in the corpus were chosen as the classifier hyperparameters. Hyperparameters are set for every language individually. Firstly, all the types of the features except one were in the “False” mode. Secondly, the amount of this type of features was increased till the overfitting markers had appeared.

By picking out different amount of features it was possible to get the accuracy at the level of 0,81 for English, 0,75 for German, 0,79 for French and 0,75 for Portuguese. The accuracy was counted based of the average results which were got during the cross-validation. To make sure that the accuracy is “stable”, I ran the classifier with the randomly mixed data sets several times. The average estimations stayed approximately at the same level. The amount of tweets in English was the highest, the corpus includes more than 2000 English text messages which affects the accuracy positively.

The overfitting was detected by taking into account the classifier’s average accuracy based on the division of the data into 10 folds with the further usage of the cross-validation method. 

## Script description
The program uses the files de_sentiment_train.csv, en_sentiment_train.csv, fr_sentiment_train.csv and pt_sentiment_train.csv to load tweets for the further classifier learning. The function load_sentiment creates a list of tweet texts and their corresponding sentiments.
The function tweet_preprocessing processes tweets before using them for the further learning. A list of the most common features for each feature type (words, bigrams, character ngrams) is made. First, the frequency dictionary is built with the function tweet_features, and then most common features (words, bigrams, symbol n-grams) are used for feature extraction. The number of features can be set individually for each type of features.

The function get_train_and_test split the data set into train and development folds. The cross-validation method is used to check the accuracy of the classifier.

The code prints the results of the training - the accuracy of each fold and the average accuracy. Optionally, the most informative features of the classifier can be printed. Then the program uses files de_sentiment_test.csv, en_sentiment_test.csv, fr_sentiment_test.csv and pt_sentiment_test.csv and creates corresponding text files (de_test.txt, en_test.txt, fr_test.txt and pt_test.txt) in the CodaLab competition format with the function classify_file.
