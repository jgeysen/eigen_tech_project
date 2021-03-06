import re
from os import listdir
from os.path import getsize, isfile, join
from typing import List, Tuple

import nltk
import pandas as pd
from cached_property import cached_property
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

import eigen_tech_project.nlp.models  # noqa
from eigen_tech_project.nlp.processing import SentenceProcessor
from eigen_tech_project.utils.errors import (
    FileNameContainsNoNumberError,
    FileNumbersNotUniqueError,
    NoFilesInDirectoryError,
    NoInterestingSentencesError,
    NoTXTFilesInDirectoryError,
    NoTXTFilesWithContentInDirectoryError,
)
from eigen_tech_project.utils.utils import no_stdout


class InvertedIndex:
    """InvertedIndex god object. Instantiate this object with the dataset of
    text files for which one wants to construct an inverted index.

    The path to the folder containing the data is a path relative to the current working directory. This path is
    provided at initialisation of the InvertedIndex instance. The file names of the files in the
    folder will need to adhere to following rules:
    * All files to be taken into account for the index calculation, should be located in the folder.
    * All files should be .txt files
    * All file names should contain at least one unique digit or number, e.g.: file1.txt, file2.txt, file3.txt, etc.

    If any of the above conditions is not met, an appropriate and indicative errors will be thrown
    throughout the calculation of the solution.

    Args:
        path: path to the folder is relative to the current working directory, containing .txt files.
    Returns:
        The InvertedIndex god object instance
    """

    def __init__(self, path):
        self.path = path
        self.sentence_splitter = nltk.data.load("tokenizers/punkt/english.pickle")
        self.sentence_processor = SentenceProcessor
        with no_stdout():
            self.inverted_index

    def __repr__(self):
        """Returns representation of the InvertedIndex object."""
        return "{}({!r})".format(self.__class__.__name__, self.path)

    @cached_property
    def file_names(self) -> List[str]:
        """Returns the names of the files in the given directory.

        Example return:
            ["file_1.txt", ..., "file_n.txt"]

        Returns:
            List: List containing the file names in strings.
        """
        file_names = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        if not file_names:
            raise NoFilesInDirectoryError
        txt_file_names = [f for f in file_names if f.endswith(".txt")]
        if not txt_file_names:
            raise NoTXTFilesInDirectoryError
        non_empty_txt_file_names = [
            f for f in txt_file_names if getsize(join(self.path, f)) != 0
        ]
        if not non_empty_txt_file_names:
            raise NoTXTFilesWithContentInDirectoryError
        else:
            return non_empty_txt_file_names

    @cached_property
    def raw_data(self) -> List[Tuple[int, str]]:
        """Returns the ids and contents of each document in list of tuples.

        Example return:
            [(1, "The contents of the first document"), ..., (n, "The contents of the n-th document.")]

        Returns:
            List: List of tuples containing the id and contents of the files in the path given at initialisation of the
            InvertedIndex instance.
        """
        # strip non-numerical characters from file names, store as tuple:
        # Output e.g.: [("file1.txt", "1"), ("file2.txt", "2"), ..., ("fileX.txt", "X")]
        file_name_to_nr_map = [(f, re.sub("[^0-9]", "", f)) for f in self.file_names]

        # All file names should contain a digit after stripping, otherwise raise error:
        if False in [f[1].isdigit() for f in file_name_to_nr_map]:
            raise FileNameContainsNoNumberError

        # Cast the numerical characters into an int:
        # Output e.g.: [("file1.txt", 1), ("file2.txt", 2), ..., ("fileX.txt", X)]
        file_name_to_nr_map = [(f[0], int(f[1])) for f in file_name_to_nr_map]

        # All file numbers in the mapping should be unique:
        if len({f[1] for f in file_name_to_nr_map}) != len(file_name_to_nr_map):
            raise FileNumbersNotUniqueError

        # read file contents for each file in the mapping:
        file_contents = [
            (f[1], open(join(self.path, f[0]), "r").read()) for f in file_name_to_nr_map
        ]
        return sorted(file_contents, key=lambda x: x[0])

    @cached_property
    def sentences(self) -> List[Tuple[int, str]]:
        """Returns a list of tuples, each containing one sentence (in string
        format) and the corresponding file id.

        Example return:
            [(1, "First sentence of the first document."),
            (1, "Second sentence of the first document."),
            ..., (n, "m-th sentence of the n-th document.")]

        Returns:
            List: List of tuples containing the file id and the sentences for the files in the path
            given at initialisation of the InvertedIndex instance.
        """
        data = []
        for file in self.raw_data:
            # split each file in sentences:
            sentences = self.sentence_splitter.tokenize(file[1])
            # list of sentence level dictionaries:
            data.extend([(file[0], sentence) for sentence in sentences])
        return data

    @cached_property
    def processed_sentences(self) -> List[Tuple]:
        """Returns a list of tuples, each containing one sentence (in string
        format), the processed version of that sentence and the corresponding
        file id.

        Example return:
            [(1, "First sentence of the first document.", "first sentence first document"),
            (1, "Second sentence of the first document.", "second sentence second document"),
            ..., (n, "Last sentence of the last document.", "last sentence last document")]

        Returns:
            List: List of tuples containing the file id and the sentences for the files in the path
            given at initialisation of the InvertedIndex instance.
        """
        return [
            sentence + (self.sentence_processor(sentence[1]).processed_sentence,)
            for sentence in self.sentences
        ]

    @cached_property
    def count_vectorizer(self):
        """Returns instance of the sklearn's CountVectorizer class, fitted on
        the processed sentences.

        Returns:
            CountVectorizer(): instance of the sklearn's CountVectorizer class, fitted on the processed sentences.
        """
        data = [x[2] for x in self.processed_sentences]
        # only fit the data to the Countvectorizer() when there is interesting data left after processing:
        if sum([len(x) for x in data]) == 0:
            raise NoInterestingSentencesError
        else:
            return CountVectorizer().fit(data)

    @cached_property
    def document_term_matrix(self) -> csr_matrix:
        """Returns sparse document-term matrix (scipy.sparse.csr_matrix).

        The document-term matrix is of size (number of sentences x size of vocabulary) and maps for each sentence the
        occurrence of each vocabulary word.

        Returns:
            csr_matrix: sparse document-term matrix.
        """
        data = [x[2] for x in self.processed_sentences]
        return self.count_vectorizer.transform(data)

    @cached_property
    def vocabulary(self) -> List[str]:
        """Returns list of unique words that occur in the data.

        The words in this list are processed, meaning that the lemma representation of the original token is used.
        The list is ordered alphabetically and maps 1-1 to the columns of the document-term matrix.

        Example return:
            ["abandon", "ability", "abroad", ..., "zone"]

        Returns:
            List: List containing the lemmatized vocabulary.
        """
        return self.count_vectorizer.get_feature_names()

    @cached_property
    def lemma_frequencies(self) -> List:
        """Returns a list of integers, representing the frequency that each
        word occurs in the vocabulary.

        The position of each integer in this list corresponds to a lemma in the
        alphabetical vocabulary list (with the same respective position). The integers represent the frequency
        a lemma occurs in the entire corpus (across sentences and files).

        Example return:
            [4, 1, 1, ..., 2]

        Returns:
            List: List of integers, representing word frequency.
        """
        return self.document_term_matrix.sum(axis=0).tolist()[0]

    @cached_property
    def lemma_occurrences(self) -> List[List[int]]:
        """Returns the a list of lists, containing sentence ids.

        This list has a length equal to the vocabulary size. The position of each sublist in this
        list maps directly to a lemma (with the same, alphabetical position) in the vocabulary list.

        The sublist here represents a collection of the sentence ids in which a lemma occurs.
        These sentence ids correspond to rows in the sentence-term matrix.

        Example return:
            [[1, 5, 29, 84], [1], ..., [128, 356]]

        Returns:
            List: List of lists, each containing sentence ids mapping the vocabulary to the sentences.
        """
        return self.document_term_matrix.transpose().tolil().rows.tolist()

    @cached_property
    def inverted_index(self) -> List[Tuple[str, int, List[int]]]:
        """Returns a list of tuples, each containing a lemma, the total
        frequency of that lemma in the corpus and a collection of the sentence
        ids that lemma occurs in.

        A mapping between a lemma and the documents (in this case: sentences) that lemma occurs in, is called an
        inverted index.

        Example return:
            [("abandon", 2, [1, 5, 29, 84]), ..., ("zone", 2, [128, 356])]

        Returns:
            List: List of tuples, mapping vocabulary to frequency to sentence ids.
        """
        return list(
            zip(self.vocabulary, self.lemma_frequencies, self.lemma_occurrences)
        )

    def mapped_inverted_index(self, save: bool = False) -> pd.DataFrame:
        """Returns a dataframe mapping the inverted index back to the original
        sentences.

        Args:
            save: Boolean, if True, the output is saved in a .csv file in the current directory. Defaults to False.
        Returns:
            pd.DataFrame():
        """
        df_input = pd.DataFrame(
            self.processed_sentences,
            columns=["document", "sentence", "lemmatized_sentence"],
        )
        df_output = pd.DataFrame(
            self.inverted_index, columns=["lemma", "frequency", "sentences"]
        )

        # map sentence ids to document ids:
        df_output["documents"] = [
            set(df_input.iloc[x].document) for x in df_output.sentences
        ]

        # map sentence ids to the original strings:
        df_output["sentences"] = [
            df_input.iloc[x].sentence.tolist() for x in df_output.sentences
        ]

        df_output = df_output.sort_values("frequency", ascending=False)
        df_output = df_output.reset_index(drop=True)

        if save:
            df_output.to_csv("output.csv", index=False)
            return df_output

        else:
            return df_output
