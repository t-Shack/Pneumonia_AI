"""
Transfer-learning architecture: an ImageNet-pretrained backbone (DenseNet121
by default) with a lightweight classification head. Backbone choice is
pluggable via config.BACKBONE — swapping to EfficientNetB0 or ResNet50 later
is a one-line config change, not a rewrite.

Keeps the same 2-output-neuron Dense(2, activation=sigmoid/softmax) head
design as the original from-scratch comparison, so the sigmoid-vs-softmax
ablation remains a true apples-to-apples secondary study — just now sitting
on a proper backbone instead of an ad-hoc 5-block CNN.
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121, EfficientNetB0, ResNet50
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

import config

BACKBONES = {
    "densenet121": {"builder": DenseNet121, "preprocess": densenet_preprocess},
    "efficientnetb0": {"builder": EfficientNetB0, "preprocess": efficientnet_preprocess},
    "resnet50": {"builder": ResNet50, "preprocess": resnet50_preprocess},
}


def get_preprocess_fn():
    """The ImageNet-normalization function matching config.BACKBONE. Both
    training (data_pipeline.py) and the Flask app (webapp/inference.py) use
    this — selected via deployment_config.json's "preprocessing" key — so
    training and inference preprocessing can never silently drift apart."""
    return BACKBONES[config.BACKBONE]["preprocess"]


def build_backbone(input_shape=None):
    input_shape = input_shape or config.IMG_SHAPE
    spec = BACKBONES[config.BACKBONE]
    base_model = spec["builder"](
        include_top=False, weights="imagenet", input_shape=input_shape, pooling=None,
    )
    base_model.trainable = False  # frozen for phase 1 (warm-up) — see unfreeze_top_layers()

    inputs = layers.Input(shape=input_shape, name="chest_xray_input")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    return inputs, x, base_model


def build_model(activation: str, name: str = None):
    """activation: "sigmoid" or "softmax". Returns an UNCOMPILED model —
    train.py compiles it (twice: once for warm-up, once after unfreezing)."""
    assert activation in ("sigmoid", "softmax"), "activation must be sigmoid or softmax"
    inputs, features, base_model = build_backbone()
    outputs = layers.Dense(2, activation=activation, name="output")(features)
    model = models.Model(
        inputs=inputs, outputs=outputs,
        name=name or f"pneumonia_{config.BACKBONE}_{activation}",
    )
    model.base_model = base_model  # stashed handle for unfreeze_top_layers(); train-time only,
                                    # does not survive a save/load round-trip (not needed to)
    return model


def unfreeze_top_layers(model, fraction=None):
    """Phase 2: unfreeze the last `fraction` of the backbone's layers for
    fine-tuning, keeping earlier (more generic) layers frozen. Call this
    AFTER phase-1 warm-up training, then recompile with a much lower
    learning rate before continuing model.fit()."""
    fraction = config.FINE_TUNE_UNFREEZE_FRACTION if fraction is None else fraction
    base_model = model.base_model
    base_model.trainable = True

    n_layers = len(base_model.layers)
    freeze_until = int(n_layers * (1 - fraction))
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    # BatchNorm layers stay frozen even within the unfrozen region — standard
    # fine-tuning practice, since batch size 32 is too small to safely
    # re-estimate BN running statistics.
    bn_frozen = 0
    for layer in base_model.layers[freeze_until:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            bn_frozen += 1

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"Unfroze {n_layers - freeze_until} of {n_layers} backbone layers "
          f"({trainable_count} actually trainable after re-freezing {bn_frozen} BatchNorm layers).")
    return model


if __name__ == "__main__":
    m = build_model("sigmoid")
    m.summary()
    print(f"\nTotal params: {m.count_params():,}")
    unfreeze_top_layers(m)
