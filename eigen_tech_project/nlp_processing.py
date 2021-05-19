import nltk
import requests
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

common_words = requests.get(
    "https://gist.githubusercontent.com/jgeysen/05a0e601396125604eaf9b99934ba0d4/raw/0ed5f860ebaef388f82b6e1c42282cec91c661de/1-1000.txt"
).text.split()


class SentenceProcessor:
    """split the document into sentences and tokenize each sentence."""

    def __init__(self, sentence):
        self.sentence = sentence
        self.tokenizer = nltk.RegexpTokenizer(r"\w+")
        self.lemmatizer = Lemmatizer()
        self.stopwords = nltk.corpus.stopwords.words("english")
        self.common_words = common_words

    def __repr__(self):
        """Returns representation of the DataLoader object."""
        return "{}({!r})".format(self.__class__.__name__, self.sentence)

    @property
    def tokenized_sentence(self):
        """Return a list of sublists with tokens."""
        return self.tokenizer.tokenize(self.sentence.lower())

    @property
    def lemmatized_sentence(self):
        """Return a list of sublists with lemmas."""
        return self.lemmatizer.lemmas(self.tokenized_sentence)

    def remove_stopwords(self, text):
        """return WORDNET POS compliance to WORDNET lemmatization (a,n,r,v)"""
        noise = set(self.stopwords + self.common_words)
        return [w for w in text if w not in noise and w.isalpha()]

    @property
    def lemmatized_sentence_no_stop(self):
        """Return a list of sublists with interesting lemmas.

        Ergo: stopword and non-alphabetical removal.
        """
        return self.remove_stopwords(self.lemmatized_sentence)

    @property
    def processed_sentence(self):
        """Return a single sentence."""
        return " ".join(self.lemmatized_sentence_no_stop)


class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __repr__(self):
        """Returns representation of the DataLoader object."""
        return "{}({!r})".format(self.__class__.__name__, "")

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """return WORDNET POS compliance to WORDNET lemmatization (a,n,r,v)"""
        if treebank_tag.startswith("J"):
            return wordnet.ADJ
        elif treebank_tag.startswith("V"):
            return wordnet.VERB
        elif treebank_tag.startswith("N"):
            return wordnet.NOUN
        elif treebank_tag.startswith("R"):
            return wordnet.ADV
        else:
            # As default part of speech tag:
            return wordnet.NOUN

    def get_lemma(self, word_postag_combo):
        """return WORDNET POS compliance to WORDNET lemmatization (a,n,r,v)"""
        return self.lemmatizer.lemmatize(
            word_postag_combo[0], self.get_wordnet_pos(word_postag_combo[1])
        )

    def lemmas(self, tokenized_sentence):
        """return WORDNET POS compliance to WORDNET lemmatization (a,n,r,v)"""
        # find the pos tagging for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = nltk.pos_tag(tokenized_sentence)
        # lemmatization using pos tags
        lemmas = [self.get_lemma(word_tag_combo) for word_tag_combo in pos_tokens]
        return lemmas
