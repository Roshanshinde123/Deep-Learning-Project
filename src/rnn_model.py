"""
Simple Recurrent Neural Network (Simple RNN) Architecture for ICU Mortality Prediction
Processes multivariate hourly time-series (N, T, 74) sequentially with static fusion (N, 9).
"""

import keras
from keras import layers, models, optimizers, metrics


def build_simple_rnn_model(
    temporal_shape: tuple = (48, 74),
    static_shape: tuple = (9,),
    learning_rate: float = 0.001,
    rnn_units: int = 64,
    dense_units_temporal: int = 32,
    dense_units_fc: int = 64,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Constructs a two-branch Simple RNN architecture:
    - Branch 1: Temporal sequential features (N, T, 74). Processes hourly observations
      chronologically (t=0 to t=T-1), updating a hidden recurrence state h_t at each step:
        h_t = tanh(W_x x_t + W_h h_{t-1} + b)
      Captures sequential accumulation of clinical deterioration across hours.
    - Branch 2: Static admission features (N, 9) projected through a dense layer.
    Both representations are concatenated into a multimodal vector feeding the sigmoid head.
    """
    # 1. Temporal Input Branch
    temporal_input = layers.Input(shape=temporal_shape, name="temporal_input")
    x_temp = layers.SimpleRNN(
        units=rnn_units,
        return_sequences=False,
        dropout=dropout_rate,
        kernel_initializer="glorot_uniform",
        name="simple_rnn_layer",
    )(temporal_input)
    x_temp = layers.Dense(dense_units_temporal, activation="relu", name="rnn_dense")(x_temp)

    # 2. Static Input Branch
    static_input = layers.Input(shape=static_shape, name="static_input")
    x_static = layers.Dense(16, activation="relu", name="static_dense")(static_input)

    # 3. Concatenation and Multimodal Classification Head
    concat = layers.Concatenate(name="multimodal_concat")([x_temp, x_static])
    x = layers.Dense(dense_units_fc, activation="relu", name="fc_dense")(concat)
    x = layers.Dropout(dropout_rate, name="fc_dropout")(x)
    output = layers.Dense(1, activation="sigmoid", name="mortality_output")(x)

    model = models.Model(
        inputs=[temporal_input, static_input],
        outputs=output,
        name="ICU_Simple_RNN_Mortality_Predictor",
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            metrics.AUC(name="roc_auc", curve="ROC"),
            metrics.AUC(name="pr_auc", curve="PR"),
            metrics.BinaryAccuracy(name="accuracy"),
        ],
    )
    return model
