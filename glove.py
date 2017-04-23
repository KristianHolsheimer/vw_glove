#!/usr/bin/python
import errno
import numpy as np
import os
import pandas as pd
import re
import sys

from csv import QUOTE_NONE
from bs4 import BeautifulSoup
from collections import Iterable, defaultdict
from itertools import islice
from nltk import sent_tokenize, regexp_tokenize
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from zipfile import ZipFile


REGEX_VW_ERROR = re.compile(r"^(?:vw|warning)(?:\s\(.+\))?\:", re.UNICODE)
REGEX_EXTRACT_TOKENS = re.sub(
    r'(?:\#.*\n|[\n\s]+)', r'',  # clean regex (remove comments and whitespace)

    r'''
        (?:
                # abbreviations
                (?:\w\.){2,}
            |
                # words
                [\w\-\/]+        # words incl. hyphens
                (?:\'t|\'s)?     # apostrophe suffixes
            |
                # numbers
                [\-\+]?          # sign
                \d+              # base digits
                (?:\.\d+)?       # decimal digits
                (?:e[\-\+]\d+)?  # scientific notation
            |
                # non-word chars
                [^\s\w]+
        )
    '''
)


class NotFittedError(Exception):
    pass


class VowpalWabbitError(Exception):
    pass


class GloVe:
    """
    Main class for computing GloVe word embeddings.

    """
    def __init__(self, n_dimensions=10):
        self.n_dimensions = n_dimensions

    def get_cooccurrence_info(self, docs):
        """
        Get the co-occurrence information for a given sequence of documents.

        Params
        ------
        docs : sequence
            Documents may consist of multiple sentences. If the docs aren't
            yet tokenized, ``nltk.sent_tokenize`` is used to split each doc
            into sentences. In turn, if the sententences aren't yet tokenized,
            ``nltk.regexp_tokenize`` is used to split each sentence into
            tokens.

        Modifies
        --------
        self : GloVe object
            Creates (or replaces) the following learned attributes:

                n_tokens_ : int
                    Number of distinct tokens seen.

                token_dictionary_ : pandas.Series
                    Maps tokens to token indices.

                token_dictionary_inv_ : pandas.Series
                    Maps token indices to tokens.

                token_counts_ : pandas.Series
                    Maps token indices to their occurrence counts.

                cooccurrence_counts_ : pandas.Series
                    Maps pairs of token indices to their co-occurrence counts.

        """
        n_tokens = [0]  # use a list for mutability

        def _index_factory():
            """ Returns a new index """
            n_tokens[0] += 1
            return n_tokens[0] - 1

        token_dictionary = defaultdict(_index_factory)
        token_counts = defaultdict(lambda: 0)
        cooccurrence_counts = defaultdict(lambda: 0)

        for doc in docs:
            if isinstance(doc, basestring):
                sents = sent_tokenize(doc)
            elif isinstance(doc, Iterable):
                sents = doc
            else:
                raise TypeError(
                    "Unexpected document type: {}".format(type(doc)))

            for sent in sents:
                if isinstance(sent, basestring):
                    tokens = regexp_tokenize(sent, REGEX_EXTRACT_TOKENS)
                elif isinstance(sent, Iterable):
                    tokens = sent
                else:
                    raise TypeError(
                        "Unexpected sentence type: {}".format(type(sent)))

                for skip, t1 in enumerate(tokens, 1):
                    i1 = token_dictionary[t1]
                    token_counts[i1] += 1
                    for t2 in tokens[skip:]:
                        i2 = token_dictionary[t2]
                        key = (i1, i2) if i1 <= i2 else (i2, i1)  # assume symm
                        cooccurrence_counts[key] += 1

        # some final processing before exposing data to user
        self.n_tokens_ = n_tokens[0]

        self.token_dictionary_ = pd.Series(
            token_dictionary, name='token_id').to_frame()
        self.token_dictionary_.index.name = 'token'

        self.token_dictionary_inv_ = (
            self.token_dictionary_
            .reset_index()
            .set_index('token_id'))

        self.token_counts_ = pd.Series(
            token_counts, name='occurrence').to_frame()
        self.token_counts_.index.name = 'token_id'

        self.cooccurrence_counts_ = pd.Series(
            cooccurrence_counts, name='cooccurrence').to_frame()
        self.cooccurrence_counts_.index.names = 'token_id1', 'token_id2'

    def save_cooccurrence_info(self, filepath, overwrite=False):
        """
        Store co-occurrence info to a zip file.

        Params
        ------
        filepath : str
            Local filepath used for storing zip file.

        overwrite : bool (default=False)
            Whether to overwrite pre-existing file.

        Modifies
        --------
        zip file on disk

        """
        filepath = filepath if filepath.endswith('.zip') else filepath + '.zip'
        if os.path.isfile(filepath) and not overwrite:
            raise IOError(
                errno.EEXIST,
                "File '{}' already exists; consider using the overwrite option"
                .format(filepath))

        cooccurrence_info = (
            ('token_counts', self.token_counts_),
            ('token_dictionary', self.token_dictionary_),
            ('token_dictionary_inv', self.token_dictionary_inv_),
            ('cooccurrence_counts', self.cooccurrence_counts_),
        )

        with ZipFile(filepath, 'w') as z:
            for name, series in cooccurrence_info:
                with NamedTemporaryFile(prefix=name, suffix='.csv') as f:
                    series.to_csv(f)
                    f.flush()
                    z.write(f.name, arcname=(name + '.csv'))

    def load_cooccurrence_info(self, filepath):
        """
        Load previously stored co-occurrence info from zip file.

        Params
        ------
        filepath : str
            Path to the co-occurrence info zip file.

        Modifies
        --------
        self : GloVe object
            Creates (or replaces) the following learned attributes:

                n_tokens_ : int
                    Number of distinct tokens seen.

                token_dictionary_ : pandas.Series
                    Maps tokens to token indices.

                token_dictionary_inv_ : pandas.Series
                    Maps token indices to tokens.

                token_counts_ : pandas.Series
                    Maps token indices to their occurrence counts.

                cooccurrence_counts_ : pandas.Series
                    Maps pairs of token indices to their co-occurrence counts.
        """
        cooccurrence_info_names = (
            ('token_counts', 'token_id'),
            ('token_dictionary', 'token'),
            ('token_dictionary_inv', 'token_id'),
            ('cooccurrence_counts', ('token_id1', 'token_id2')),
        )
        with ZipFile(filepath) as z:
            for name, index in cooccurrence_info_names:
                filename = name + '.csv'
                attrname = name + '_'
                with z.open(filename) as f:
                    setattr(self, attrname, pd.read_csv(f, index_col=index))

        self.n_tokens_ = self.token_dictionary_.shape[0]

    def get_top_n_tokens(self, n=10):
        """
        Get the top-n most frequently occurring tokens.

        Params
        ------
        n : int (default=10)
            Number of tokens to return

        Returns
        -------
        top_n : pandas.Series
            List of top-n tokens with their respective occurrence counts.

        """
        return (
            self.token_counts_
            .sort_values('occurrence', ascending=False)
            .iloc[:n]
            .join(self.token_dictionary_inv_, how='inner')
            .sort_values('occurrence', ascending=False))

    def get_top_n_cooccurrences(self, n=10):
        """
        Get the top-n most frequently co-occurring tokens.

        Params
        ------
        n : int (default=10)
            Number of tokens to return

        Returns
        -------
        top_n : pandas.Series
            List of top-n tokens with their respective occurrence counts.

        """
        return (
            self.cooccurrence_counts_
            .sort_values('cooccurrence', ascending=False)
            .iloc[:n]
            .reset_index()
            .merge(
                self.token_dictionary_inv_.rename(columns={'token': 'token1'}),
                left_on='token_id1', right_index=True)
            .merge(
                self.token_dictionary_inv_.rename(columns={'token': 'token2'}),
                left_on='token_id2', right_index=True)
            .set_index(['token_id1', 'token_id2'])
            .sort_values('cooccurrence', ascending=False)
        )

    def vw_lines(self, shuffle=True, random_state=None, truncate=None):
        """
        Generator that yields Vowpal Wabbit formatted labeled example lines.

        Params
        ------
        shuffle : bool (default=True)
            Whether to shuffle to data before iterating through it.

        truncate : int (default=None)
            Truncate the generator after

        """
        if not hasattr(self, 'cooccurrence_counts_'):
            raise NotFittedError(
                "No co-occurrence info available; please run either "
                "`get_cooccurrence_info` or `load_cooccurrence_info`")

        vw_template = "{0:f} {1:f} |u {2:d} |v {3:d}"
        series = self.cooccurrence_counts_.iloc[:truncate, 0]
        if shuffle:
            series = series.sample(frac=1, random_state=random_state)

        def fudge_factor(x, alpha=0.75, x_max=10):
            """ GloVe heuristic sample weights """
            if x > x_max:
                return 1.0
            return pow(x / float(x_max), alpha)

        for (i, j), x in series.iteritems():
            y = np.log10(x)
            w = fudge_factor(x)
            yield vw_template.format(y, w, i, j)
            if i == j:
                continue
            yield vw_template.format(y, w, j, i)

    def vw_lines_diagonal_only(self, shuffle=True, truncate=None):
        """
        Generator that yields Vowpal Wabbit formatted labeled example lines.

        Params
        ------
        shuffle : bool (default=True)
            Whether to shuffle to data before iterating through it.

        truncate : int (default=None)
            Truncate the generator after

        """
        if not hasattr(self, 'token_dictionary_'):
            raise NotFittedError(
                "No co-occurrence info available; please run either "
                "`get_cooccurrence_info` or `load_cooccurrence_info`")

        vw_template = "|u {0:d} |v {0:d}"

        for i in islice(self.token_dictionary_inv_.index, truncate):
            yield vw_template.format(i)

    def _train_word_vectors(self, model_filepath):

        with NamedTemporaryFile(prefix='glove_traindata_', suffix='.vw') as f:

            for line in self.vw_lines():
                f.write(line + "\n")

            f.flush()
            vw = Popen(['vw', f.name, '-f', model_filepath,
                        '--hash', 'strings', '-q', 'uv',
                        '--rank', str(self.n_dimensions)], stderr=PIPE)

            for line in vw.stderr:
                if REGEX_VW_ERROR.match(line):
                    raise VowpalWabbitError(line.strip())
                sys.stderr.write(line)
                sys.stderr.flush()

            vw.wait()

    def _extract_word_vectors(self, model_filepath):
        latent_factors = {}

        with NamedTemporaryFile(prefix='glove_testdata_', suffix='.vw') as f:

            for line in self.vw_lines_diagonal_only():
                f.write(line + "\n")

            f.flush()
            vw = Popen(['vw', f.name, '-t', '-i', model_filepath,
                        '--audit', '--quiet'], stdout=PIPE)

            for line in vw.stdout:
                if not line.startswith('\t'):
                    continue
                _get_latent_factors(line, latent_factors)

            vw.wait()

        return latent_factors

    def compute_word_vectors(self, refresh=False):
        """
        Compute GloVe word embeddings from co-occurrence info.

        Params
        ------
        refresh : bool (default=False)
            Whether to re-compute word vectors in case they are already
            available.

        """
        if hasattr(self, 'word_vectors_') or refresh:
            print >> sys.stderr, (
                "Word vectors are already available; set refresh=True if you "
                "wish to re-compute the word vectors.")
            return

        with NamedTemporaryFile(prefix='glove_model_', suffix='.vw') as f:
            self._train_word_vectors(f.name)
            latent_factors = self._extract_word_vectors(f.name)

        latent_factors = pd.Series(latent_factors)
        latent_factors.index.names = 'token_id', 'vector_index'
        latent_factors.sort_index(inplace=True)

        latent_factors = latent_factors.unstack(level='vector_index')
        self.word_vectors = (
            latent_factors.join(self.token_dictionary_inv_).set_index('token'))

    def save(self, filepath_or_buffer):
        """
        Store word embeddings to disk.

        Params
        ------
        filepath_or_buffer : str
            Any object that ``pandas.to_csv()`` can write to.

        """
        formatted = self.word_vectors_word_vectors.applymap("{:.6f}".format)
        formatted.to_csv(
            filepath_or_buffer, sep=' ', header=None, quoting=QUOTE_NONE)

    @classmethod
    def load(cls, filepath_or_buffer):
        """
        Load word embeddings from a csv file.

        Params
        ------
        filepath_or_buffer : str
            Any object that ``pandas.to_csv()`` can read from.

        """
        word_vectors = pd.read_csv(
            filepath_or_buffer, sep=' ',
            quoting=QUOTE_NONE, index_col=0, header=None)
        word_vectors.columns = np.arange(word_vectors.shape[1])
        word_vectors.index.name = 'token'
        self = cls(n_dimensions=word_vectors.shape[1])
        self.word_vectors_ = word_vectors
        return self


def _get_latent_factors(line, latent_factors):
    line = line.strip().split('\t')[3:]
    for info in line:
        info = info.split(':')
        i, token_id = info[0].split('^')

        i = int(i[1:]) - 1        # vector index of latent factors
        token_id = int(token_id)  # token_id
        u = float(info[3])        # vector entry of u factor
        v = float(info[7])        # vector entry of v factor

        # update to latest known values
        latent_factors[(token_id, i)] = (u + v) / 2.0


def _clean_strings_iter(seq):
    for string in seq:
        string = BeautifulSoup(string, "html.parser").get_text()  # remove html
        string = string.encode('ascii', errors='ignore')          # only ascii
        string = string.lower()                                   # lowercase
        yield string


def main():
    # TODO: use ArgumentParser here
    glove = GloVe(n_dimensions=10)
    glove.get_cooccurrence_info(_clean_strings_iter(sys.stdin))
    glove.save_cooccurrence_info('cooccurrence_info.zip', overwrite=False)
    glove.compute_word_vectors()
    glove.save('glove_vectors.csv')


if __name__ == '__main__':
    main()
