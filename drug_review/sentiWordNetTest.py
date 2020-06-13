from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

synsets = wn.synsets('cheerful', pos=None)
print("Synsets:")
print(synsets)
synet = synsets[0]
print("Synet name:")
print(synet.name())
print("Synet definition: ")
print(synet.definition())
print("Synet examples: ")
print(synet.examples())

swn_synset = swn.senti_synset(synet.name())
word_sent = swn_synset.pos_score() - swn_synset.neg_score()

print("Positive score = "+ str(swn_synset.pos_score()))
print("Negative score = "+ str(swn_synset.neg_score()))
print("Sentiment = "+ str(word_sent))