# #-------------------------------DEV NOTES
# [] Clean the data set
# [X] So we need pos tagging and understand sentences by words if it is verb,........
# [X] #Replacing Negations with Antonyms (We can add 'never' as well)
# [] Then make senti_synsets and create new column with it
# [] Also do not forget to create your own functions to get this shitty functions/methods
# [] We can use other methods in cookbook to clean the text
# #-------------------------------------------------------------------------------------------
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk import sent_tokenize, word_tokenize, pos_tag
import re
lemmatizer = WordNetLemmatizer()

#-------------------------------------------------------------FUNCTIONS-------------------------------------------------
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
    print("Raw_sentences:")
    print(raw_sentences)
    sentiment = 0
    tokens_count = 0

    for raw_sentence in raw_sentences:
        raw_sentence = replace_negations(raw_sentence) #Replacing Negations with Antonyms (Python 3 Text Processing with NLTK 3 Cookbook)
        tagged_sentence = pos_tag(word_tokenize(raw_sentence))
        print("Tagged_sentence:")
        print(tagged_sentence)

        for word, tag in tagged_sentence:
            wn_tag = penn_to_wn(tag)
            if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
                continue
            print("Wn_tag:")
            print(wn_tag)

            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            if not lemma:
                continue
            print("Lemma:")
            print(lemma)

            synsets = wn.synsets(lemma, pos=wn_tag)
            if not synsets:
                continue
            print("Synets:")
            print(synsets)

            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            word_sent = swn_synset.pos_score() - swn_synset.neg_score()

            if word_sent != 0:
                sentiment += word_sent
                tokens_count += 1

    if tokens_count == 0:
        return 0
    sentiment = sentiment/tokens_count
    print("Sentiment = "+ str(sentiment))
    if sentiment >= 0.01:
        return 1
    if sentiment <= -0.01:
        return -1
    return 0

#----------------------------------------------------------END: FUNCTIONS-----------------------------------------------
#----------------------------------------------READ DATA / CREATE TRAIN & TEST SETS-------------------------------------

# # print(os.listdir("resources/drugReviewRawData"))
# #
# # train = pd.read_csv('resources/drugReviewRawData/rawTrain.csv')
# # print(train.head(5))
# # print(train.shape)
# # print(train["review"][:10])
#-------------------------------------------END: READ DATA / CREATE TRAIN & TEST SETS-----------------------------------

# input = "If I could give it a 0, I would absolutely do so.  Started at 50mg, and felt WIRED.  Wanted to get up and clean the house!  Bumped it to 100mg, less wired, but still wide awake all night.  Bumped to 150, with the same lack of effect.  MD informed me after this dose it becomes less effective for sleep, so why even bother.  15 years of trying different sleep medications and alternatives, and this, I can say for sure, was the LEAST effective I have ever come across.  At it&#039;s low price point, feel free to give it a try, and maybe you will be luckier?  Everyone&#039;s sleep conditions are different.  But if you get hyper after benadryl, expect the same reaction to this drug."
# print(sentiment_sentiwordnet(input))

#print(replace_negations('MD informed me after this dose it becomes less effective for sleep, so why even bother.'))

print(sentiment_sentiwordnet('it is effective.'))
print('---------------------------------------------------------------------------------')
print(sentiment_sentiwordnet('it is less effective.'))
