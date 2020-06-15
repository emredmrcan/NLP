# #-----------------------------------------------Libraries and Packages------------------------------------------------
import re
from time import process_time

import nltk
import pandas as pd
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')
corpus_words = set(nltk.corpus.words.words())
stop_words = set(stopwords.words('english'))
# #-----------------------------------------END: Libraries and Packages-------------------------------------------------
# #----------------------------------------------FUNCTIONS--------------------------------------------------------------
def clean_data(data):
    data = data[~data.condition.str.contains(" users found this comment helpful.", na=False)]
    data = data[data.duplicated('condition', keep=False)]  # Remove unique conditions
    data["review"] = data["review"].apply(clean_reviews)
    return data


def clean_reviews(raw_review):
    letters_only = re.sub('[^a-zA-Z]', ' ', raw_review)
    meaningful_words = " ".join(
        w for w in nltk.wordpunct_tokenize(letters_only) if w.lower() in corpus_words or not w.isalpha())
    return meaningful_words


def create_sentiment_column(data):
    start = process_time()
    print("--------CREATE SENTIMENT STARTED----------")
    data["sentiment"] = data["review"].apply(sentiment_sentiwordnet)
    end = process_time()
    print("Elapsed time for creating the sentiment column in seconds:",end-start)
    print("-----------CREATE SENTIMENT END-----------")
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
    i, j = 0, len(text)
    words = []

    while i < j:
        word = text[i]

        if word == 'not' and i + 1 < j:
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
    return wn.NOUN


def sentiment_sentiwordnet(text):
    raw_sentences = sent_tokenize(text)

    sentiment = 0
    tokens_count = 0

    for raw_sentence in raw_sentences:
        raw_sentence = replace_negations(
            raw_sentence)  # Replacing Negations with Antonyms (Python 3 Text Processing with NLTK 3 Cookbook)
        tokenizedWords = word_tokenize(raw_sentence)
        stopWordsRemoved = [w for w in tokenizedWords if not w in stop_words]
        tagged_sentence = pos_tag(stopWordsRemoved)

        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue

            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            word_sent = swn_synset.pos_score() - swn_synset.neg_score()

            if word_sent != 0:
                sentiment += word_sent
                tokens_count += 1

    if tokens_count == 0:
        return 0
    sentiment = sentiment / tokens_count

    if sentiment >= 0.01:
        return 1
    if sentiment <= -0.01:
        return -1
    return 0


def normalize_ratings(row):
    if row.rating > 9:
        return 1
    elif row.rating < 3:
        return -1
    else:
        return 0


def readTrainData(path):
    start = process_time()
    print("------------READ DATA STARTED-------------")
    train = pd.read_csv(path,usecols = ['drugName','condition','review','rating']).dropna(how = 'any', axis = 0)
    end = process_time()
    print("Elapsed time for reading the train data in seconds:",end-start)
    print("--------------READ DATA END---------------")
    return train


def cleanData(data):
    start = process_time()
    print("------------CLEAN DATA STARTED------------")
    cleanedData = clean_data(data)
    end = process_time()
    print("Elapsed time for cleaning the train data in seconds:", end - start)
    print("---------------CLEAN DATA END-------------")
    return cleanedData

def result_based_sentiwordnet_with_prepared_sentiment_result(testSet):
    pred_y = testSet["sentiment"]
    accuracy = accuracy_score(testSet.normalized_ratings, pred_y)
    print("Accuracy for sentiwordnet result:")
    print(accuracy)
    print(confusion_matrix(testSet.normalized_ratings, pred_y))
    print(classification_report(testSet.normalized_ratings, pred_y) )


def result_based_sentiwordnet(testSet):
    pred_y = testSet["review"].apply(sentiment_sentiwordnet)
    accuracy = accuracy_score(testSet.normalized_ratings, pred_y)
    print("Accuracy for sentiwordnet result:")
    print(accuracy)


# #----------------------------------------------------------END: FUNCTIONS---------------------------------------------
# #----------------------------------------------------------MAIN-------------------------------------------------------

# #----------- Run with Raw Data
# # Reading data: 0.7 sn   Cleaning data: 12 sn Sentiment Analysis: 8.5 min. In local machine.
train = readTrainData('../resources/drugReviewRawData/rawTrain.csv')
cleanedTrainData = cleanData(train)
trainDataWithSentiment = create_sentiment_column(cleanedTrainData).drop(columns="review")
trainDataWithSentiment = pd.DataFrame(trainDataWithSentiment, columns=['rating','sentiment'])

# #----------- Run with the data that is already cleaned and acquired sentiment results
# trainDataWithSentiment = pd.read_csv('../resources/cleanedData/CleanedTrainWithSentiment.csv', usecols=['rating', 'sentiment']).dropna(how='any', axis=0)



trainDataWithSentiment["normalized_ratings"] = trainDataWithSentiment.apply(normalize_ratings, axis=1)
trainDataWithSentiment = trainDataWithSentiment.drop(['rating'], axis=1)
trainDataWithSentiment.drop(trainDataWithSentiment[trainDataWithSentiment.normalized_ratings == 0].index,inplace=True)
trainDataWithSentiment.drop(trainDataWithSentiment[trainDataWithSentiment.sentiment == 0].index,inplace=True)
trainSet = trainDataWithSentiment.sample(frac=0.8, random_state=200)  # random state is a seed value
testSet = trainDataWithSentiment.drop(trainSet.index)

print("result_based_sentiwordnet_with_prepared_sentiment_result started")
result_based_sentiwordnet_with_prepared_sentiment_result(testSet)

