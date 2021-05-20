from eigen_tech_project.nlp_processing import Lemmatizer

# from eigen_tech_project.nlp_processing import SentenceProcessor


def test_lemmatizer_lemmas():
    tokenized_sentence = ["hello", "I", "am", "an", "engineer"]
    lemmatizer = Lemmatizer()

    lemmas = lemmatizer.lemmas(tokenized_sentence)
    lemmas_exp = ["hello", "I", "be", "an", "engineer"]

    assert lemmas == lemmas_exp


def test_lemmatizer_get_wordnet_pos():
    treebank_tags = [
        "CC",
        "CD",
        "DT",
        "EX",
        "FW",
        "IN",
        "JJ",
        "JJR",
        "JJS",
        "LS",
        "MD",
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "PDT",
        "POS",
        "PRP",
        "PRP$",
        "RB",
        "RBR",
        "RBS",
        "RP",
        "SYM",
        "TO",
        "UH",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        "WDT",
        "WP",
        "WP$",
        "WRB",
    ]

    equivalent_wordnet_tags = [
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "a",
        "a",
        "a",
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "r",
        "r",
        "r",
        "r",
        "n",
        "n",
        "n",
        "v",
        "v",
        "v",
        "v",
        "v",
        "v",
        "n",
        "n",
        "n",
        "n",
    ]

    lemmatizer = Lemmatizer()
    equivalent_wordnet_tags_exp = [
        lemmatizer.get_wordnet_pos(treebank_tag) for treebank_tag in treebank_tags
    ]

    assert equivalent_wordnet_tags == equivalent_wordnet_tags_exp


def test_sentence_processor():
    pass
