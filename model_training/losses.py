"""
Categorical focal loss (Lin et al., 2017) — down-weights easy,
high-confidence-correct examples during training so the model can't reach
good overall accuracy just by nailing the majority class. This is the
current standard fix for class-imbalanced classification, and a more
targeted tool than class_weight alone for the exact failure mode this
project ran into (model defaulting to predicting PNEUMONIA).

IMPORTANT: any script that calls tensorflow.keras.models.load_model() on a
model trained with this loss must `import losses` first (even if unused
directly) — the @register_keras_serializable decorator only takes effect
once this module has been imported, which is what lets load_model()
reconstruct the loss by name instead of raising "Unknown loss function".
"""

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="pneumonia_project")
class CategoricalFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="categorical_focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg


def get_loss(loss_name: str, gamma: float = 2.0, alpha: float = 0.25):
    """loss_name: "focal" or "categorical_crossentropy" — see config.LOSS_FUNCTION."""
    if loss_name == "focal":
        return CategoricalFocalLoss(gamma=gamma, alpha=alpha)
    if loss_name == "categorical_crossentropy":
        return "categorical_crossentropy"
    raise ValueError(f"Unknown loss_name: {loss_name!r}")
