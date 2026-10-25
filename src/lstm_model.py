"""
Long Short-Term Memory (LSTM) Architecture for ICU Mortality Prediction
Primary deep learning architecture utilizing gated cell states to track multi-day patient trajectories.
"""

import keras
from keras import layers, models, optimizers, metrics


def build_lstm_model(
    temporal_shape: tuple = (48, 74),
    static_shape: tuple = (9,),
    learning_rate: float = 0.001,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dropout_between_lstm: float = 0.3,
    dense_units_fc: int = 64,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Constructs the primary two-branch LSTM architecture:
    - Branch 1: Temporal sequential features (N, T, 74). Utilizes memory cells governed
      by three gates (forget gate f_t, input gate i_t, output gate o_t):
        f_t = sigmoid(W_f x_t + U_f h_{t-1} + b_f)
        i_t = sigmoid(W_i x_t + U_i h_{t-1} + b_i)
        C_t = f_t * C_{t-1} + i_t * tanh(W_c x_t + U_c h_{t-1} + b_c)
        o_t = sigmoid(W_o x_t + U_o h_{t-1} + b_o)
        h_t = o_t * tanh(C_t)
      The additive cell state gradient flow eliminates vanishing gradients across 48 time steps,
      enabling retention of critical early admission signals alongside late decompensation trends.
    - Branch 2: Static admission features (N, 9) projected through a dense layer.
    Both representations are concatenated into a multimodal vector feeding the sigmoid head.
    """
    # 1. Temporal Input Branch
    temporal_input = layers.Input(shape=temporal_shape, name="temporal_input")
    x_temp = layers.LSTM(
        units=lstm_units_1,
        return_sequences=True,
        kernel_initializer="glorot_uniform",
        name="lstm_layer_1",
    )(temporal_input)
    x_temp = layers.Dropout(dropout_between_lstm, name="lstm_dropout")(x_temp)
    x_temp = layers.LSTM(
        units=lstm_units_2,
        return_sequences=False,
        kernel_initializer="glorot_uniform",
        name="lstm_layer_2",
    )(x_temp)

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
        name="ICU_LSTM_Mortality_Predictor",
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
