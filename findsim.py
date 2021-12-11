#!/usr/bin/env python3
"""
Determine most similar words in terms of their word embeddings.
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import numpy as np
from torch.nn import functional as F

from integerize import Integerizer   # look at integerize.py for more info

# For type annotations, which enable you to check correctness of your code:
from typing import List, Optional

try:
    import torch as th
    import torch.nn as nn
except ImportError:
    print("\nERROR! Try installing Miniconda and activating it.\n")
    raise


log = logging.getLogger(Path(__file__).stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("embeddings", type=Path, help="Path to word embeddings file.")
    parser.add_argument("word", type=str, help="Word to lookup")
    parser.add_argument("--minus", type=str, default=None)
    parser.add_argument("--plus", type=str, default=None)

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()
    if not args.embeddings.is_file():
        parser.error("You need to provide a real file of embeddings.")
    if (args.minus is None) != (args.plus is None):  # != is the XOR operation!
        parser.error("Must include both of `plus` and `minus` or neither.")

    return args

class Lexicon:
    """
    Class that manages a lexicon and can compute similarity.

    >>> my_lexicon = Lexicon.from_file(my_file)
    >>> my_lexicon.find_similar_words(bagpipe)
    """

    def __init__(self, words, values) -> None:
        self.lexicon_words = words
        self.lexicon_values = values.astype(np.float64)
        """Load information into coupled word-index mapping and embedding matrix."""

    @classmethod
    def from_file(cls, file: Path) -> Lexicon:
        # FINISH THIS FUNCTION

        with open(file) as f:
            next(f)
            lexicon_values = []
            lexicon_words = []
            for line in f:  # All of the other lines are regular.
                line = line.split()
                lexicon_words.append(line.pop(0))
                lexicon_values.append(line)
            lexicon_values = np.array(lexicon_values)


        lexicon = Lexicon(words=lexicon_words, values=lexicon_values)
        return lexicon

    def find_similar_words(
        self, word: str, *, plus: Optional[str] = None, minus: Optional[str] = None
    ) -> List[str]:
        """Find most similar words, in terms of embeddings, to a query."""

        if (minus is None) != (plus is None):
            raise TypeError("Must include both of `plus` and `minus` or neither.")
        # Keep going!

        word_index = self.lexicon_words.index(word)
        word_values = self.lexicon_values[word_index]
        self.lexicon_words.pop(word_index)
        self.lexicon_values = np.delete(self.lexicon_values, word_index, 0)

        if minus is not None:
            if not word == minus:
                minus_index = self.lexicon_words.index(minus)
                minus_values = self.lexicon_values[minus_index]
                self.lexicon_values = np.delete(self.lexicon_values, minus_index, 0)
                self.lexicon_words.pop(minus_index)
            else:
                minus_values = word_values
            if not word == plus:
                if not minus == plus:
                    plus_index = self.lexicon_words.index(plus)
                    plus_values = self.lexicon_values[plus_index]
                    self.lexicon_values = np.delete(self.lexicon_values, plus_index, 0)
                    self.lexicon_words.pop(plus_index)
                else:
                    plus_values = minus_values
            else:
                plus_values = word_values
            word_values = word_values - minus_values + plus_values
        output = F.cosine_similarity(th.FloatTensor(self.lexicon_values), th.FloatTensor(word_values), dim=-1)
        top_values, top_indices = th.topk(th.FloatTensor(output), 10, dim=-1)
        top_words = [self.lexicon_words[i] for i in top_indices.tolist()]

        return top_words


def format_for_printing(word_list: List[str]) -> str:
    return " ".join(word_list)


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    lexicon = Lexicon.from_file(args.embeddings)
    similar_words = lexicon.find_similar_words(
        args.word, plus=args.plus, minus=args.minus
    )
    print(format_for_printing(similar_words))


if __name__ == "__main__":
    main()
