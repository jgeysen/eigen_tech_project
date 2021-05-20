import nltk

try:  # pragma: no cover
    nltk.data.load("tokenizers/punkt/english.pickle")
except LookupError:  # pragma: no cover
    nltk.download("punkt")

try:  # pragma: no cover
    nltk.corpus.stopwords.words("english")
except LookupError:  # pragma: no cover
    nltk.download("stopwords")

try:  # pragma: no cover
    nltk.data.load(
        "taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle"
    )
except LookupError:  # pragma: no cover
    nltk.download("averaged_perceptron_tagger")

try:  # pragma: no cover
    from nltk.corpus import wordnet  # noqa

    x = wordnet.NOUN
except LookupError:  # pragma: no cover
    nltk.download("wordnet")
