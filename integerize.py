#!/usr/bin/env python3

from typing import (Dict, Generic, Iterable, Iterator, List, Optional, TypeVar,
                    overload)

T = TypeVar("T")  # see https://mypy.readthedocs.io/en/stable/generics.html


class Integerizer(Generic[T]):
    """
    A collection of distinct object types, such as a vocabulary or a set of parameter names,
    that are associated with consecutive ints starting at 0.

    Example usage:

    >>> from integerize import Integerizer
    >>> vocab: Integerizer[str]                       # type hint saying that the objects will be strings
    >>> vocab = Integerizer(['','hello','goodbye'])   # lets the empty string '' be 0
    >>> vocab.index('goodbye')                        # convert from word to int
    2
    >>> vocab[2]                                      # convert from int back to word
    'goodbye'
    >>> sentence = ('hello','world','if','world','you','be')
    >>> [vocab.index(w) for w in sentence]    # convert from words to ints (or None if OOV)
    [1, None, None, None, None, None]
    >>> [vocab.index(w, add=True) for w in sentence]  # expand vocabulary on demand (so no OOVs)
    [1, 3, 4, 3, 5, 6]
    >>> [vocab[i] for i in [1, 3, 4, 3, 5, 6]]        # convert from ints back to words
    ['hello', 'world', 'if', 'world', 'you', 'be']
    >>> len(vocab)                                    # vocab size (not including OOV)
    7
    >>> vocab[:]    # show all 7 word types, in order of their ints
    ['', 'hello', 'goodbye', 'world', 'if', 'you', 'be']
    >>> 'world' in vocab, 'mars' in vocab
    (True, False)
    """

    # If you are unfamiliar with the special __ method names, check out
    # https://docs.python.org/3/reference/datamodel.html#special-method-names .

    def __init__(self, iterable=[]):
        """
        Initialize the collection to the empty set, or to the set of *unique* objects in its argument
        (in order of first occurrence).
        """
        # Set up a pair of data structures to convert objects to ints and back again.
        self._objects: List[T] = []       # list of all unique objects that have been added so far
        self._indices: Dict[T, int] = {}  # maps each object to its integer position in the list
        # Add any objects that were given.
        self.update(iterable)

    def __eq__(self, other) -> bool:
        return self._objects == other._objects

    def __len__(self) -> int:
        """
        Number of objects in the collection.
        """
        return len(self._objects)

    def __iter__(self) -> Iterator[T]:
        """
        Iterate over all the objects in the collection.
        """
        return iter(self._objects)

    def __contains__(self, obj: T) -> bool:
        """
        Does the collection contain this object?  (Implements `in`.)
        """
        return self.index(obj) is not None

    @overload  # To appease the type checker and provide better hints in your editor!
    def __getitem__(self, index: int) -> T:
        ...  # We'll get overridden by the implemented version below.

    @overload
    def __getitem__(self, index: slice) -> List[T]:
        ...  # This is to type-hint the fact that we support slices: my_integerizer[3:5]

    def __getitem__(self, index):
        """
        Return the object with a given index.  
        (Implements subscripting, e.g., `my_integerizer[3]` and `my_integerizer[3:5]`.)
        """
        return self._objects[index]

    def index(self, obj: T, add=False) -> Optional[int]:
        """
        The integer associated with a given object, or `None` if the object is not in the collection (OOV).  
        Use `add=True` to add the object if it is not present. 
        """
        try:
            return self._indices[obj]
        except KeyError:
            if not add:
                return None

            # add the object to both data structures
            i = len(self)
            self._objects.append(obj)
            self._indices[obj] = i
            return i

    def add(self, obj: T) -> None:
        """
        Add the object if it is not already in the collection.
        Similar to `set.add` (or `list.append`).
        """
        self.index(obj, add=True)  # call for side effect; ignore return value

    def update(self, iterable: Iterable[T]):
        """
        Add all the objects if they are not already in the collection.
        Similar to `set.update` (or `list.extend`).
        """
        for obj in iterable:
            self.add(obj)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
