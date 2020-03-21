# #-------------------------------DEV NOTES
# [] Clean the data set
# [X] So we need pos tagging and understand sentences by words if it is verb,........
# [X] #Replacing Negations with Antonyms (We can add 'never' as well)
# [] Then make senti_synsets and create new column with it
# [] Also do not forget to create your own functions to get this shitty functions/methods
# [] We can use other methods in cookbook to clean the text
# #-------------------------------------------------------------------------------------------
import os
import nltk
import pandas as pd
import re

from time import process_time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')
corpus_words = set(nltk.corpus.words.words())


# #-------------------------------------------------------------FUNCTIONS-------------------------------------------------


def clean_data(data):
    data = data[~data.condition.str.contains(" users found this comment helpful.", na=False)]
    data = data[data.duplicated('condition', keep=False)]
    data["review"] = data["review"].apply(clean_reviews)
    return data


def clean_reviews(raw_review):
    letters_only = re.sub('[^a-zA-Z]', ' ', raw_review)
    # print('Letters only:' + letters_only)
    meaningful_words = " ".join(w for w in nltk.wordpunct_tokenize(letters_only) if w.lower() in corpus_words or not w.isalpha())
    # print('Meaningful words:' + meaningful_words)
    return meaningful_words


def create_sentiment_column(data):
    # data["sentiment"] = data["review"].swifter.apply(sentiment_sentiwordnet)
    data["sentiment"] = data["review"].apply(sentiment_sentiwordnet)
    return data


def replace(word, pos=None):
    antonyms = set()

    for syn in wn.synsets(word, pos=pos):
        for lemma in syn.lemmas():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())

    if len(antonyms) == 1:
        return antonyms.pop()
    else:
        return None


def replace_negations(text):
    text = re.findall(r'\w+', text)
    i, l = 0, len(text)
    words = []

    while i < l:
        word = text[i]

        if word == 'not' and i + 1 < l:
            ant = replace(text[i + 1])

            if ant:
                words.append(ant)
                i += 2
                continue

        words.append(word)
        i += 1

    return ' '.join(words)


def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def sentiment_sentiwordnet(text):
    raw_sentences = sent_tokenize(text)
    # print("Raw_sentences:")
    # print(raw_sentences)
    sentiment = 0
    tokens_count = 0

    for raw_sentence in raw_sentences:
        raw_sentence = replace_negations(raw_sentence) #Replacing Negations with Antonyms (Python 3 Text Processing with NLTK 3 Cookbook)
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
        # print("Tagged_sentence:")
        # print(tagged_sentence)

        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
            # print("Wn_tag:")
            # print(wn_tag)

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            # print("Lemma:")
            # print(lemma)

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
            # print("Synets:")
            # print(synsets)

            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            word_sent = swn_synset.pos_score() - swn_synset.neg_score()

            if word_sent != 0:
                sentiment += word_sent
                tokens_count += 1

    if tokens_count == 0:
        return 0
    sentiment = sentiment/tokens_count
    # print("Sentiment = "+ str(sentiment))
    if sentiment >= 0.01:
        return 1
    if sentiment <= -0.01:
        return -1
    return 0


def normalize_ratings(row):
    if row.rating > 7:
        return 1
    elif row.rating < 4:
        return -1
    else :
        return 0


def result_based_sentiwordnet(testSet):
    pred_y = testSet["review"].apply(sentiment_sentiwordnet)
    accuracy = accuracy_score(testSet.normalized_ratings, pred_y)
    print("Accuracy for sentiwordnet result:")
    print(accuracy)


def result_based_automated_sentiment_analyzer(trainSet, testSet):
    clf = Pipeline([('vectorizer', CountVectorizer(analyzer="word", ngram_range=(1, 2),
                                                   tokenizer=word_tokenize, max_features=10000)),
                    ('classifier', LinearSVC())])

    clf = clf.fit(trainSet.review, trainSet.normalized_ratings)
    score = clf.score(testSet.review, testSet.normalized_ratings)
    print("Accuracy for automated sentiment analyzer result:")
    print(score)
# #----------------------------------------------------------END: FUNCTIONS---------------------------------------------
# #----------------------------------------------READ DATA / CREATE TRAIN & TEST SETS-----------------------------------

# print(os.listdir("resources/drugReviewRawData"))
# start = process_time()
# print("------------READ DATA STARTED-------------")
# train = pd.read_csv('resources/drugReviewRawData/rawTrain.csv',usecols = ['drugName','condition','review','rating']).dropna(how = 'any', axis = 0)
# # train = pd.read_csv('CleanedTrainWithoutSentiment.csv')
# end = process_time()
# print("Elapsed time for reading the train data in seconds:",end-start)
# print("--------------READ DATA END---------------")
#
# start = process_time()
# print("------------CLEAN DATA STARTED------------")
# train = clean_data(train)
# end = process_time()
# print("Elapsed time for cleaning the train data in seconds:",end-start)
# print("---------------CLEAN DATA END-------------")
#
# train.to_csv('CleanedTrainWithoutSentiment.csv', index=False)
#
#
# start = process_time()
# print("--------CREATE SENTIMENT STARTED----------")
# train = create_sentiment_column(train).drop(columns="review")
# end = process_time()
# print("Elapsed time for creating the sentiment column in seconds:",end-start)
# print("-----------CREATE SENTIMENT END-----------")
#
# train.to_csv('CleanedTrainWithSentiment.csv', index=False)
# print(train.shape)

# start = process_time()
# print("------------READ CleanedTrainWithSentiment DATA STARTED-------------")
# train = pd.read_csv('CleanedTrainWithSentiment.csv',usecols = ['rating','sentiment']).dropna(how = 'any', axis = 0)
# end = process_time()
# print("Elapsed time for reading the train data in seconds:",end-start)
# print("--------------READ CleanedTrainWithSentiment DATA END---------------")

start = process_time()
print("------------READ CleanedTrainWithoutSentiment DATA STARTED-------------")
train = pd.read_csv('CleanedTrainWithoutSentiment.csv',usecols = ['review','rating']).dropna(how = 'any', axis = 0)
end = process_time()
print("Elapsed time for reading the train data in seconds:",end-start)
print("--------------READ CleanedTrainWithoutSentiment DATA END---------------")

train["normalized_ratings"] = train.apply(normalize_ratings, axis=1)
train = train.drop(['rating'], axis = 1)
trainSet=train.sample(frac=0.8,random_state=200) #random state is a seed value
testSet=train.drop(trainSet.index)

# #-----------------------------------------END: READ DATA / CREATE TRAIN & TEST SETS-----------------------------------

result_based_sentiwordnet(testSet)
result_based_automated_sentiment_analyzer(trainSet,testSet)

