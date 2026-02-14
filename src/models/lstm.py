"""LSTM model definition for turning point classification."""

import logging
from typing import Tuple

from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout

logger = logging.getLogger(__name__)


def build_lstm_model(
    input_shape: Tuple[int, int],
    hidden_units: int = 200,
) -> Sequential:
    """Build and compile a binary-classification LSTM model.

    Architecture: Input → LSTM → Dropout → Dense(1, sigmoid).

    Args:
        input_shape: ``(window_size, n_features)``.
        hidden_units: Number of LSTM hidden units.

    Returns:
        Compiled Keras Sequential model.
    """
    logger.info(
        "Building LSTM model: input_shape=%s, hidden_units=%d",
        input_shape,
        hidden_units,
    )
    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(hidden_units, dropout=0.2, recurrent_dropout=0.2),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
