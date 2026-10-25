"""
1D Convolutional Neural Network (1D CNN) Architecture for ICU Mortality Prediction
Integrates temporal time-series channel (N, T, 74) and static admission channel (N, 9).
"""

import keras
from keras import layers, models, optimizers, metrics


def build_cnn_model(
    temporal_shape: tuple = (48, 74),
    static_shape: tuple = (9,),
    learning_rate: float = 0.001,
    conv_filters_1: int = 64,
    conv_filters_2: int = 128,
    kernel_size: int = 3,
    dense_units: int = 64,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Constructs a two-branch 1D CNN architecture:
    - Branch 1: Temporal sequential features (N, T, 74). Applies 1D convolutions across
      time steps to detect localized physiological patterns (e.g., sudden acute hypoxemia,
      transient hypotensive dips, rapid heart rate spikes).
    - Branch 2: Static admission features (N, 9) processed through a dense projection layer.
    Both representations are concatenated and passed to classification head with sigmoid output.
    """
    # 1. Temporal Input Branch
    temporal_input = layers.Input(shape=temporal_shape, name="temporal_input")
    x_temp = layers.Conv1D(
        filters=conv_filters_1,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name="conv1d_1",
    )(temporal_input)
    x_temp = layers.BatchNormalization(name="bn_1")(x_temp)
    x_temp = layers.Activation("relu", name="relu_1")(x_temp)
    x_temp = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x_temp)

    x_temp = layers.Conv1D(
        filters=conv_filters_2,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name="conv1d_2",
    )(x_temp)
    x_temp = layers.BatchNormalization(name="bn_2")(x_temp)
    x_temp = layers.Activation("relu", name="relu_2")(x_temp)
    x_temp = layers.GlobalAveragePooling1D(name="global_avg_pool")(x_temp)

    # 2. Static Input Branch
    static_input = layers.Input(shape=static_shape, name="static_input")
    x_static = layers.Dense(16, activation="relu", name="static_dense")(static_input)

    # 3. Merged Multimodal Representation
    concat = layers.Concatenate(name="multimodal_concat")([x_temp, x_static])
    x = layers.Dense(dense_units, activation="relu", name="fc_dense")(concat)
    x = layers.Dropout(dropout_rate, name="fc_dropout")(x)
    output = layers.Dense(1, activation="sigmoid", name="mortality_output")(x)

    model = models.Model(
        inputs=[temporal_input, static_input],
        outputs=output,
        name="ICU_1D_CNN_Mortality_Predictor",
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
