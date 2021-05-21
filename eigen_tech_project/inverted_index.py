import re
from os import listdir
from os.path import abspath, isfile, join
from typing import List, Tuple

import nltk
import pandas as pd
from cached_property import cached_property
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

import eigen_tech_project.nlp_models  # noqa
from eigen_tech_project.nlp_processing import SentenceProcessor
from eigen_tech_project.utils import no_stdout


class InvertedIndex:
    """Returns representation of the DataLoader object."""

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
        return [f for f in listdir(self.path) if isfile(join(self.path, f))]

    @cached_property
    def raw_data(self) -> List[Tuple[int, str]]:
        """Returns the ids and contents of each document in list of tuples.

        Example return:
            [(1, "The contents of the first document"), ..., (n, "The contents of the n-th document.")]

        Returns:
            List: List of tuples containing the id and contents of the files in the path given at initialisation of the
            InvertedIndex instance.
        """
        return [
            (
                int(re.sub("[^0-9]", "", f)),
                open(abspath(join(self.path, f)), "r").read(),
            )
            for f in self.file_names
        ]

    @cached_property
    def sentences(self) -> List[Tuple[int, str]]:
        """Returns a list of tuples, each containing one sentence (in string
        format) and the corresponding file id.

        Example return:
            [(1, "First sentence of the first document."), (1, "Second sentence of the first document."), ..., (n, "m-th sentence of the n-th document.")]

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
            [(1, "First sentence of the first document.", "first sentence first document"), (1, "Second sentence of the first document.", "second sentence second document"), ..., (n, "Last sentence of the last document.", "last sentence last document")]

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
        return CountVectorizer().fit(data)

    @cached_property
    def document_term_matrix(self) -> csr_matrix:
        """Returns sparse matrix (scipy.sparse.csr_matrix) of size (number of
        sentences x size of vocabulary), mapping for each sentence the
        occurrence of each word in the vocabulary.

        Returns:
            csr_matrix: sparse document-term matrix.
        """
        data = [x[2] for x in self.processed_sentences]
        return self.count_vectorizer.transform(data)

    @cached_property
    def vocabulary(self) -> List:
        """Returns the contents of each document in list of tuples.

        Example return:
            [(1, "The contents of the first document"), ..., (n, "The contents of the n-th document.")]

        Returns:
            List: Tupled list containing the id and contents of the documents in path given at InvertedIndex
            initialisation.
        """
        return self.count_vectorizer.get_feature_names()

    @cached_property
    def lemma_frequencies(self) -> List:
        """Returns the contents of each document in list of tuples.

        Example return:
            [(1, "The contents of the first document"), ..., (n, "The contents of the n-th document.")]

        Returns:
            List: Tupled list containing the id and contents of the documents in path given at InvertedIndex
            initialisation.
        """
        return self.document_term_matrix.sum(axis=0).tolist()[0]

    @cached_property
    def lemma_occurrences(self) -> List[List]:
        """Returns the contents of each document in list of tuples.

        Example return:
            [(1, "The contents of the first document"), ..., (n, "The contents of the n-th document.")]

        Returns:
            List: Tupled list containing the id and contents of the documents in path given at InvertedIndex
            initialisation.
        """
        return self.document_term_matrix.transpose().tolil().rows.tolist()

    @cached_property
    def inverted_index(self) -> List[Tuple[str, int, set]]:
        """Returns the contents of each document in list of tuples.

        Example return:
            [(1, "The contents of the first document"), ..., (n, "The contents of the n-th document.")]

        Returns:
            List: Tupled list containing the id and contents of the documents in path given at InvertedIndex
            initialisation.
        """
        return list(
            zip(self.vocabulary, self.lemma_frequencies, self.lemma_occurrences)
        )

    def mapped_inverted_index(self, save=True):
        """Returns the contents of each document in list of tuples.

        Example return:
            [(1, "The contents of the first document"), ..., (n, "The contents of the n-th document.")]

        Returns:
            List: Tupled list containing the id and contents of the documents in path given at InvertedIndex
            initialisation.
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
