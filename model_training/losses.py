"""
Categorical focal loss (Lin et al., 2017) — down-weights easy,
high-confidence-correct examples so the model can't reach good overall
accuracy just by nailing the majority class.

BUG FIX #1 (earlier): renormalizes y_pred before computing the loss —
matching what Keras's built-in categorical_crossentropy does internally.
Without this, a 2-neuron SIGMOID head never receives gradient pressure
pushing either neuron DOWN — only ever up. See the long comment in git
history / model_training/README.md for the full diagnosis.

BUG FIX #2 (this version): alpha is now per-class, not a single flat
scalar. The original paper's alpha is written alpha_t specifically because
it's meant to take a DIFFERENT value depending on the true class — usually
higher for the rare class — to directly counteract class imbalance. This
implementation previously used one alpha=0.25 for every example regardless
of class, which only provided the "focus on hard examples" benefit (via
gamma) and NONE of the class-balancing benefit alpha is supposed to add.
Confirmed in practice: NORMAL recall was only ~61% on the primary test set
and the model needed a post-hoc threshold correction to reach ~90%+ — a
class-balanced alpha directly targets the same root cause the threshold
fix works around. See train.py's compute_focal_alpha().

IMPORTANT: any script that calls tensorflow.keras.models.load_model() on a
model trained with this loss must `import losses` first (even if unused
directly) — the @register_keras_serializable decorator only takes effect
once this module has been imported.
"""

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="pneumonia_project")
class CategoricalFocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, name="categorical_focal_loss", **kwargs):
        """
        alpha: either a single float (applied uniformly — the old
        behavior, kept for comparison/ablation via
        config.FOCAL_ALPHA_STRATEGY = "fixed") or a list/tuple of one alpha
        per class in config.CLASS_NAMES order (the class-balanced fix,
        config.FOCAL_ALPHA_STRATEGY = "class_balanced", the new default).
        """
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha  # kept as-given (float or list) for get_config()/serialization
        if isinstance(alpha, (list, tuple)):
            self._alpha_tensor = tf.constant(alpha, dtype=tf.float32)
        else:
            self._alpha_tensor = tf.constant(float(alpha), dtype=tf.float32)  # broadcasts to every class

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        # Couples independent-sigmoid outputs the same way Keras's built-in
        # categorical_crossentropy does internally. No-op for softmax.
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1.0 - keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred)
        # alpha_tensor is either scalar (broadcasts to all classes, old
        # behavior) or shape (num_classes,) (broadcasts per-class against
        # y_true's one-hot encoding, picking out that example's true-class alpha).
        weight = self._alpha_tensor * y_true * tf.pow(1.0 - y_pred, self.gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg


def get_loss(loss_name: str, gamma: float = 2.0, alpha=0.25):
    """loss_name: "focal" or "categorical_crossentropy" — see config.LOSS_FUNCTION.
    alpha: float or per-class list — see CategoricalFocalLoss docstring."""
    if loss_name == "focal":
        return CategoricalFocalLoss(gamma=gamma, alpha=alpha)
    if loss_name == "categorical_crossentropy":
        return "categorical_crossentropy"
    raise ValueError(f"Unknown loss_name: {loss_name!r}")
