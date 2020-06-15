from nltk.corpus import wordnet, stopwords
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))


def getDetails(word):
    syns = wordnet.synsets(word)
    print("Synsets: {}".format(syns))
    for syn in syns:
        print("Synset name:" + syn.name())
        print("Synset lemmas: {}".format(syn.lemmas()))
        print("Synset definition: " + syn.definition())
        print("Synset examples: {}".format(syn.examples()))
        printSWNresult(syn.name())

def printSWNresult(synetName) :
    swn_synset = swn.senti_synset(synetName)
    word_sent = swn_synset.pos_score() - swn_synset.neg_score()
    print("---SWN results----")
    print("Positive score = " + str(swn_synset.pos_score()))
    print("Negative score = " + str(swn_synset.neg_score()))
    print("Sentiment = " + str(word_sent))


def getRelatedTermsOfWord(word):
    syns = wordnet.synsets(word)
    syn = syns[0]
    print(syn.lemmas())
    print(syn.hypernyms())
    print(syn.hyponyms())
    print(syn.member_holonyms())
    print(syn.part_meronyms())

getDetails("wonder")
getRelatedTermsOfWord("dog")

example_sent = "This is a sample sentence, showing off the stop words filtration."
word_tokens = word_tokenize(example_sent)
filtered_sentence = [w for w in word_tokens if not w in stop_words]
print("Word tokens: {}".format(word_tokens))
print("Filtered sentence: {}".format(filtered_sentence))