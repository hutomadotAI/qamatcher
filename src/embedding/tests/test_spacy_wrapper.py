# flake8: noqa

import pytest
import embedding.spacy_wrapper


def test_tokenizing_1():
    spacy_wrapper = embedding.spacy_wrapper.SpacyWrapper()
    result = spacy_wrapper.tokenizeSpacy("hi")
    assert len(result) == 1
    assert result[0] == "hi"
