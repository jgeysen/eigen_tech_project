from eigen_tech_project.nlp_processing import Lemmatizer


def test_lemmatizer_lemmas():
    tokenized_sentence = ["hello", "I", "am", "an", "engineer"]
    lemmatizer = Lemmatizer()

    lemmas = lemmatizer.lemmas(tokenized_sentence)
    goal_lemmas = ["hello", "I", "be", "an", "engineer"]

    for i in range(0, len(lemmas)):
        assert lemmas[i] == goal_lemmas[i]


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

    for i in range(0, len(treebank_tags)):
        assert (
            lemmatizer.get_wordnet_pos(treebank_tags[i]) == equivalent_wordnet_tags[i]
        )


def test_sentence_processor():
    pass
