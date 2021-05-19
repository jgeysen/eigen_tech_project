import nltk

# import the nlp language model:
try:  # pragma: no cover
    nltk.data.load("tokenizers/punkt/english.pickle")
    # nltk_data = nltk.data.load("corpora/wordnet.zip/wordnet/")
except LookupError:  # pragma: no cover
    # nltk.download("wordnet")
    nltk.download("punkt")

try:  # pragma: no cover
    stopwords = nltk.corpus.stopwords.words("english")
except LookupError:  # pragma: no cover
    nltk.download("stopwords")
