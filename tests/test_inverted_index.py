import pandas as pd
from pandas._testing import assert_frame_equal

from eigen_tech_project.inverted_index import InvertedIndex


def test_inverted_index(tmp_path):
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

    ii = InvertedIndex(path=d)
    assert ii.file_names == ["test_file1.txt"]

    raw_data_exp = [(1, content)]
    assert ii.raw_data == raw_data_exp

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

    frequencies_exp = [1] * 7
    assert ii.lemma_frequencies == frequencies_exp

    lemma_occurrence_exp = [[0], [4], [2], [1], [3], [0], [0]]
    assert ii.lemma_occurrences == lemma_occurrence_exp

    inverted_index_exp = list(
        zip(vocabulary_exp, frequencies_exp, lemma_occurrence_exp)
    )
    assert ii.inverted_index == inverted_index_exp

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

    df_exp = pd.DataFrame(
        list(zip(lemma, frequency, sentences, documents)),
        columns=["lemma", "frequency", "sentences", "documents"],
    )

    assert_frame_equal(df_exp, ii.mapped_inverted_index(save=False))
