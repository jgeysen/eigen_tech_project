import re
from os import listdir
from os.path import abspath, isfile, join

import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from eigen_tech_project.nlp_processing import SentenceProcessor


class InvertedIndex:
    def __init__(self, path):
        self.path = path
        self.sentence_splitter = nltk.data.load("tokenizers/punkt/english.pickle")
        self.sentence_processor = SentenceProcessor

    def __repr__(self):
        """Returns representation of the DataLoader object."""
        return "{}({!r})".format(self.__class__.__name__, self.path)

    @property
    def file_names(self):
        """Returns representation of the DataLoader object."""
        return [f for f in listdir(self.path) if isfile(join(self.path, f))]

    @property
    def raw_data(self):
        """Returns representation of the DataLoader object."""
        return [
            (
                int(re.sub("[^0-9]", "", f)),
                open(abspath(join(self.path, f)), "r").read(),
            )
            for f in self.file_names
        ]

    @property
    def sentences(self):
        """Returns representation of the DataLoader object."""
        data = []
        for file in self.raw_data:
            # split each file in sentences:
            sentences = self.sentence_splitter.tokenize(file[1])
            # list of sentence level dictionaries:
            data.extend([(file[0], sentence) for sentence in sentences])
        return data

    @property
    def processed_sentences(self):
        """Returns representation of the DataLoader object."""
        return [
            sentence + (self.sentence_processor(sentence[1]).processed_sentence,)
            for sentence in self.sentences
        ]

    @property
    def count_vectorizer(self):
        """Return list with the top x most occuring interesting words, with
        following elements: (feature_id, occurence)."""
        data = [x[2] for x in self.processed_sentences]
        return CountVectorizer().fit(data)

    @property
    def document_term_matrix(self):
        """Return list with the top x most occuring interesting words, with
        following elements: (feature_id, occurence)."""
        data = [x[2] for x in self.processed_sentences]
        return self.count_vectorizer.transform(data)

    @property
    def vocabulary(self):
        """Return list with the top x most occuring interesting words, with
        following elements: (feature_id, occurence)."""
        return self.count_vectorizer.get_feature_names()

    @property
    def lemma_frequencies(self):
        """Return list with the top x most occuring interesting words, with
        following elements: (feature_id, occurence)."""
        return self.document_term_matrix.sum(axis=0).tolist()[0]

    @property
    def lemma_occurences(self):
        """Return list with the top x most occuring interesting words, with
        following elements: (feature_id, occurence)."""
        occurences = [
            self.document_term_matrix[:, i].nonzero()[0].tolist()
            for i in range(0, len(self.vocabulary))
        ]
        return occurences

    @property
    def inverted_index(self):
        """Return list with the top x most occuring interesting words, with
        following elements: (feature_id, occurence)."""
        return list(zip(self.vocabulary, self.lemma_frequencies, self.lemma_occurences))

    @property
    def solution(self):
        df_input = pd.DataFrame(self.sentences, columns=["document", "sentence"])
        df_output = pd.DataFrame(
            self.inverted_index, columns=["lemma", "frequency", "sentences"]
        )

        # map sentence ids to documents:
        df_output["documents"] = [
            set(df_input.iloc[x].document) for x in df_output.sentences
        ]

        # map sentence ids to strings:
        df_output["sentences"] = [
            df_input.iloc[x].sentence.tolist() for x in df_output.sentences
        ]

        df_output = df_output.sort_values("frequency", ascending=False).reset_index(
            drop=True
        )
        df_output.to_csv("output.csv", index=False)

        return df_output
