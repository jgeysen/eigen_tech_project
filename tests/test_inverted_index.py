import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from eigen_tech_project.inverted_index import InvertedIndex
from eigen_tech_project.utils.errors import (
    FileNameContainsNoNumberError,
    FileNumbersNotUniqueError,
    NoFilesInDirectoryError,
    NoInterestingSentencesError,
    NoTXTFilesInDirectoryError,
    NoTXTFilesWithContentInDirectoryError,
)


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


def test_InvertedIndex_not_a_directory(tmp_path):
    # given ...
    # ... a mocked path containing a folder called "test_data"
    # ... in this folder a mocked file called "test_file1.txt" containing a text with 7 sentences.
    d = tmp_path / "test_data"
    # when ... we create an InvertedIndex object for this mocked path:
    with pytest.raises(FileNotFoundError):
        InvertedIndex(path=d)


def test_InvertedIndex_no_files_in_directory(tmp_path):
    # given ...
    # ... a mocked path containing a folder called "test_data"
    # ... in this folder a mocked file called "test_file1.txt" containing a text with 7 sentences.
    d = tmp_path / "test_data"
    d.mkdir()
    # when ... we create an InvertedIndex object for this mocked path:
    with pytest.raises(NoFilesInDirectoryError):
        InvertedIndex(path=d)


def test_InvertedIndex_no_txt_files_in_directory(tmp_path):
    """Test the InvertedIndex class."""
    # given ...
    # ... a mocked path containing a folder called "test_data"
    # ... in this folder a mocked file called "test_file1.txt" containing a text with 7 sentences.
    d = tmp_path / "test_data"
    d.mkdir()
    p = d / "test_file1.json"
    content = """{
              "Name": "Test",
              "Mobile": 12345678,
              "Boolean": True,
              "Pets": ["Dog", "cat"],
              "Address": {
                "Permanent address": "USA",
               "current Address": "AU"
              }
            }"""
    p.write_text(content)

    # when ... we create an InvertedIndex object for this mocked path:
    with pytest.raises(NoTXTFilesInDirectoryError):
        InvertedIndex(path=d)


def test_InvertedIndex_only_empty_txt_files_in_directory(tmp_path):
    """Test the InvertedIndex class."""
    # given ...
    # ... a mocked path containing a folder called "test_data"
    # ... in this folder a mocked file called "test_file1.txt" containing a text with 7 sentences.
    d = tmp_path / "test_data"
    d.mkdir()
    p1 = d / "test_file1.txt"
    content = ""
    p1.write_text(content)
    p2 = d / "test_file2.txt"
    content = ""
    p2.write_text(content)

    # when ... we create an InvertedIndex object for this mocked path:
    with pytest.raises(NoTXTFilesWithContentInDirectoryError):
        InvertedIndex(path=d)


def test_InvertedIndex_file_numbers_not_unique(tmp_path):
    """Test the InvertedIndex class."""
    # given ...
    # ... a mocked path containing a folder called "test_data"
    # ... in this folder a mocked file called "test_file1.txt" containing a text with 7 sentences.
    d = tmp_path / "test_data"
    d.mkdir()
    p1 = d / "test_file1.txt"
    content = (
        "Let me begin by saying thanks to all you who've traveled, from far and wide, "
        "to brave the cold today. We all made this journey for a reason. It's humbling, "
        "but in my heart I know you didn't come here just for me, you came here because "
        "you believe in what this country can be."
    )
    p1.write_text(content)
    p2 = d / "test_data1.txt"
    content = (
        "In the face of war, you believe there can be peace. In the face of despair, "
        "you believe there can be hope. But let me tell you how I came to be here."
    )
    p2.write_text(content)
    # when ... we create an InvertedIndex object for this mocked path:
    with pytest.raises(FileNumbersNotUniqueError):
        InvertedIndex(path=d)


def test_InvertedIndex_file_name_no_number(tmp_path):
    """Test the InvertedIndex class."""
    # given ...
    # ... a mocked path containing a folder called "test_data"
    # ... in this folder a mocked file called "test_file1.txt" containing a text with 7 sentences.
    d = tmp_path / "test_data"
    d.mkdir()
    p1 = d / "test_file.txt"
    content = (
        "Let me begin by saying thanks to all you who've traveled, from far and wide, "
        "to brave the cold today. We all made this journey for a reason. It's humbling, "
        "but in my heart I know you didn't come here just for me, you came here because "
        "you believe in what this country can be."
    )
    p1.write_text(content)
    # when ... we create an InvertedIndex object for this mocked path:
    with pytest.raises(FileNameContainsNoNumberError):
        InvertedIndex(path=d)


def test_InvertedIndex_no_interesting_content_in_files(tmp_path):
    """Test the InvertedIndex class."""
    # given ...
    # ... a mocked path containing a folder called "test_data"
    # ... in this folder a mocked file called "test_file1.txt" containing a text with 7 sentences.
    d = tmp_path / "test_data"
    d.mkdir()
    p1 = d / "test_file1.txt"
    content = "The of to and a in is it you. That he was for on are with as I his they."
    p1.write_text(content)
    p2 = d / "test_file2.txt"
    content = (
        "Come did number sound no most people my over know water than call first who may. "
        "down side been now find any new work part take get place made live where. After "
        "back little only round man year came show every good."
    )
    p2.write_text(content)
    # when ... we create an InvertedIndex object for this mocked path:
    with pytest.raises(NoInterestingSentencesError):
        InvertedIndex(path=d)
