class NoFilesInDirectoryError(Exception):
    def __init__(self):
        self.errmsg = "There are no files in the provided directory."

    def __str__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.errmsg)


class NoTXTFilesInDirectoryError(Exception):
    def __init__(self):
        self.errmsg = "There are no .txt-files in the provided directory."

    def __str__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.errmsg)


class NoTXTFilesWithContentInDirectoryError(Exception):
    def __init__(self):
        self.errmsg = "The .txt files in the directory are empty."

    def __str__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.errmsg)


class FileNameContainsNoNumberError(Exception):
    def __init__(self):
        self.errmsg = "One or more file names contain no numbers."

    def __str__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.errmsg)


class FileNumbersNotUniqueError(Exception):
    def __init__(self):
        self.errmsg = (
            "The numbers embedded in the file names are not unique for each file."
        )

    def __str__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.errmsg)


class NoInterestingSentencesError(Exception):
    def __init__(self):
        self.errmsg = "The text(s) provided in the .txt files contains only stopwords and/or common words."

    def __str__(self):
        return "<{}: {}>".format(self.__class__.__name__, self.errmsg)
