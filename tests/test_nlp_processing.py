from eigen_tech_project.nlp.nlp_processing import Lemmatizer, SentenceProcessor


def test_Lemmatizer_lemmas():
    """Test the InvertedIndex class."""
    # given ...
    # ... a tokenized sentence: ["hello", "I", "am", "an", "engineer"]
    tokenized_sentence = ["hello", "I", "am", "an", "engineer"]
    # ... an instance of the Lemmatizer class:
    lemmatizer = Lemmatizer()

    assert isinstance(lemmatizer, Lemmatizer)

    # then ..
    # ... the lemmas property should contain a list of lemmas:
    lemmas = lemmatizer.lemmas(tokenized_sentence)
    lemmas_exp = ["hello", "I", "be", "an", "engineer"]
    assert lemmas == lemmas_exp


def test_Lemmatizer_get_wordnet_pos():
    """Test the InvertedIndex class."""
    # given ...
    # ... a list of all possible treebank tags and their equivalen wordnet tags:
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
    # ... an instance of the Lemmatizer class:
    lemmatizer = Lemmatizer()
    assert isinstance(lemmatizer, Lemmatizer)

    # then ..
    # ... the get_wordnet_pos attribute of the Lemmatizer class should map the treebank tags to their equivalen
    # wordnet tags:
    equivalent_wordnet_tags_exp = [
        lemmatizer.get_wordnet_pos(treebank_tag) for treebank_tag in treebank_tags
    ]

    assert equivalent_wordnet_tags == equivalent_wordnet_tags_exp


def test_SentenceProcessor():
    # given ...
    # ... a test_sentence:
    test_sentence = (
        "Let me begin by saying thanks to all you who have traveled, from far "
        "and wide, to brave the cold today."
    )
    # ... an instance of the Lemmatizer class:
    sp = SentenceProcessor(test_sentence)
    assert isinstance(sp, SentenceProcessor)

    # then ..
    # ... the sentence property should contain the original sentence:
    assert sp.sentence == test_sentence

    # ... the tokenized_sentence property should contain the tokens of the original sentence:
    assert sp.tokenized_sentence == [
        "let",
        "me",
        "begin",
        "by",
        "saying",
        "thanks",
        "to",
        "all",
        "you",
        "who",
        "have",
        "traveled",
        "from",
        "far",
        "and",
        "wide",
        "to",
        "brave",
        "the",
        "cold",
        "today",
    ]

    # ... the lemmatized_sentence property should contain the lemmas of the original sentence:
    assert sp.lemmatized_sentence == [
        "let",
        "me",
        "begin",
        "by",
        "say",
        "thanks",
        "to",
        "all",
        "you",
        "who",
        "have",
        "travel",
        "from",
        "far",
        "and",
        "wide",
        "to",
        "brave",
        "the",
        "cold",
        "today",
    ]

    # ... the lemmatized_sentence_no_stop property should contain the interesting lemmas of the original sentence:
    assert sp.lemmatized_sentence_no_stop == ["thanks", "brave", "today"]

    # ... the processed_sentence property should contain a concatenation of the lemmatized_sentence_no_stop:
    assert sp.processed_sentence == "thanks brave today"
