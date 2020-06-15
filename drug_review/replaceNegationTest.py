from nltk.corpus import wordnet as wn
import re
import sys


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


assert replace_negations("I am not happy") == 'I am unhappy', 'Should be I am unhappy'
assert replace_negations("I am not curious") == 'I am incurious', 'Should be I am incurious'
assert replace_negations("I do not love") == 'I do hate', 'Should be I do hate'
assert replace_negations("It is not wanted") == 'It is unwanted', 'Should be It is unwanted'
assert replace_negations("It is not effective") == 'It is ineffective', 'Should be It is ineffective'
print("All tests are passed!")
sys.exit()
