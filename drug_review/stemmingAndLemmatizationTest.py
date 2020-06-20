import sys
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')

assert lemmatizer.lemmatize('drugs') == 'drug', 'Should be drug'
assert stemmer.stem('using') == 'use', 'Should be use'

# Irrelavant token
assert lemmatizer.lemmatize('asdd') == 'asdd', 'Should be asdd'
assert stemmer.stem('asdd') == 'asdd', 'Should be asdd'

print("All tests are passed!")
sys.exit()