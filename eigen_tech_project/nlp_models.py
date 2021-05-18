import nltk.data

# import the nlp language model:
try:  # pragma: no cover
    nltk_pipe = nltk.data.load("tokenizers/punkt/english.pickle")
    nltk_data = nltk.data.load("corpora/wordnet.zip/wordnet/")
except LookupError:  # pragma: no cover
    nltk.download("wordnet")
    nltk.download("punkt")
    nltk_pipe = nltk.data.load("tokenizers/punkt/english.pickle")
