from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet as swn

def getDetails(word):
    syns = wordnet.synsets(word)
    print("Synsets:")
    print(*syns, sep='\n')
    for syn in syns:
        print("Synset name:")
        print(syn.name())
        print("Synset lemmas:")
        lemmas = syn.lemmas()
        print(lemmas)
        print("Synset definition:")
        print(syn.definition())
        print("Synset examples:")
        examples = syn.examples()
        print(*examples, sep='\n')
        print("SWN results:")
        printSWNresult(syn.name())

def printSWNresult(synetName) :
    swn_synset = swn.senti_synset(synetName)
    word_sent = swn_synset.pos_score() - swn_synset.neg_score()

    print("Positive score = " + str(swn_synset.pos_score()))
    print("Negative score = " + str(swn_synset.neg_score()))
    print("Sentiment = " + str(word_sent))

    # synonyms = []
    # antonyms = []
    #
    # for syn in wordnet.synsets("good"):
    #     for l in syn.lemmas():
    #         synonyms.append(l.name())
    #         if l.antonyms():
    #             antonyms.append(l.antonyms()[0].name())
    #
    # print(set(synonyms))
    # print(set(antonyms))

getDetails("wonder")