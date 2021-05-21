import pandas as pd
from pandas._testing import assert_frame_equal

from eigen_tech_project.inverted_index import InvertedIndex


def test_InvertedIndex(tmp_path):
    """Test the InvertedIndex class."""
    # given ...
    # ... a mocked path containing a folder called "test_data"
    # ... in this folder a mocked file called "test_file1.txt" containing a text with 7 sentences.
    d = tmp_path / "test_data"
    d.mkdir()
    p = d / "test_file1.txt"
    content = (
        "Let me begin by saying thanks to all you who've traveled, from far and wide, "
        "to brave the cold today. We all made this journey for a reason. It's humbling, "
        "but in my heart I know you didn't come here just for me, you came here because "
        "you believe in what this country can be. In the face of war, you believe there "
        "can be peace. In the face of despair, you believe there can be hope. But let me "
        "tell you how I came to be here."
    )
    p.write_text(content)

    # when ... we create an InvertedIndex object for this mocked path:
    ii = InvertedIndex(path=d)
    assert isinstance(ii, InvertedIndex)

    assert ii.path == d

    # then ..
    # ... the file names property should return the name of our mocked file:
    assert ii.file_names == ["test_file1.txt"]

    # then ..
    # ... the raw_data property should contain the file identifier (1) and the content string:
    raw_data_exp = [(1, content)]
    assert ii.raw_data == raw_data_exp

    # then ..
    # ... the sentences property should map to the following hardcoded sentences:
    sentences = [
        "Let me begin by saying thanks to all you who've traveled, from far and wide, "
        "to brave the cold today.",
        "We all made this journey for a reason.",
        "It's humbling, but in my heart I know you didn't come here just for me, you "
        "came here because you believe in what this country can be.",
        "In the face of war, you believe there can be peace.",
        "In the face of despair, you believe there can be hope.",
        "But let me tell you how I came to be here.",
    ]
    sentences_exp = [(1, sentence) for sentence in sentences]
    assert ii.sentences == sentences_exp

    # then ..
    # ... the processed_sentences property should contain the interesting lemmas for each sentence:
    processed_sentences_exp = [
        "thanks brave today",
        "journey",
        "humble",
        "peace",
        "despair",
        "",
    ]
    processed_sentences_exp = [
        sentences_exp[i] + (processed_sentences_exp[i],) for i in range(0, 6)
    ]
    assert ii.processed_sentences == processed_sentences_exp

    # then ..
    # ... the vocabulary property should contain an alphabetically ordered set of the interesting lemmas in
    # the entire corpus (across documents and sentences).
    vocabulary_exp = [
        "brave",
        "despair",
        "humble",
        "journey",
        "peace",
        "thanks",
        "today",
    ]
    assert ii.vocabulary == vocabulary_exp

    # then ..
    # ... the lemma_frequencies property should contain the frequency of occurence of each word in the vocabulary
    # across the entire corpus (documents and sentences).
    frequencies_exp = [1] * 7
    assert ii.lemma_frequencies == frequencies_exp

    # then ..
    # ... the lemma_occurence property should contain the sub-lists of ids of the sentences in which each word in the
    # vocabulary occurs.
    lemma_occurrence_exp = [[0], [4], [2], [1], [3], [0], [0]]
    assert ii.lemma_occurrences == lemma_occurrence_exp

    # then ..
    # ... the inverted_index property should contain a list of tuples, mapping the vocabulary, frequencies and
    # lemma_occurence:
    inverted_index_exp = list(
        zip(vocabulary_exp, frequencies_exp, lemma_occurrence_exp)
    )
    assert ii.inverted_index == inverted_index_exp

    # then ..
    # ... the mapped_inverted_index DataFrame should contain the following data:
    lemma = ["brave", "despair", "humble", "journey", "peace", "thanks", "today"]
    frequency = [1, 1, 1, 1, 1, 1, 1]
    sentences = [
        [
            "Let me begin by saying thanks to all you who've traveled, from far and wide, to brave the cold today."
        ],
        ["In the face of despair, you believe there can be hope."],
        [
            "It's humbling, but in my heart I know you didn't come here just for me, you came here because you believe in what this country can be."
        ],
        ["We all made this journey for a reason."],
        ["In the face of war, you believe there can be peace."],
        [
            "Let me begin by saying thanks to all you who've traveled, from far and wide, to brave the cold today."
        ],
        [
            "Let me begin by saying thanks to all you who've traveled, from far and wide, to brave the cold today."
        ],
    ]
    documents = [{1}, {1}, {1}, {1}, {1}, {1}, {1}]

    assert_frame_equal(
        ii.mapped_inverted_index(save=False),
        pd.DataFrame(
            list(zip(lemma, frequency, sentences, documents)),
            columns=["lemma", "frequency", "sentences", "documents"],
        ),
    )
