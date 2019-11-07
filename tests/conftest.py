import os
import pytest

from trainer.champop import CardGenerator

CARDS_PICKLE = 'data/cards.pck'

@pytest.fixture
def card_generator():
    card_gen = CardGenerator(os.path.join(os.curdir, CARDS_PICKLE))
    return card_gen
