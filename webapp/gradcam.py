"""
Grad-CAM, generated live for whatever image a user uploads. Duplicated from
model_training/gradcam.py rather than imported — the webapp is meant to be
deployable standalone, without a dependency on the training project's
source tree.
"""

import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras


def find_backbone_submodel(model):
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("No nested backbone sub-model found in this model.")


def find_last_conv_layer(backbone):
    for layer in reversed(backbone.layers):
        if len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No 4D (conv-like) layer found in backbone.")


def make_gradcam_heatmap(model, image_batch, class_index, last_conv_layer_name=None):
    backbone = find_backbone_submodel(model)
    last_conv_layer_name = last_conv_layer_name or find_last_conv_layer(backbone)

    grad_model = keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(last_conv_layer_name).output, backbone.output],
    )
    gap = model.get_layer("global_avg_pool")
    dropout = model.get_layer("head_dropout")
    output_layer = model.get_layer("output")

    with tf.GradientTape() as tape:
        conv_output, backbone_output = grad_model(image_batch)
        tape.watch(conv_output)
        x = gap(backbone_output)
        x = dropout(x, training=False)
        preds = output_layer(x)
        class_score = preds[:, class_index]

    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        raise RuntimeError("Gradient is None — check last_conv_layer_name is upstream of output.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + keras.backend.epsilon())
    return heatmap.numpy()


def overlay_heatmap(original_image_0_1, heatmap, alpha=0.4, colormap="jet"):
    import matplotlib.cm as cm

    h, w = original_image_0_1.shape[:2]
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (h, w)).numpy().squeeze()
    cmap = matplotlib.colormaps.get_cmap(colormap) if hasattr(matplotlib.colormaps, 'get_cmap') else matplotlib.colormaps[colormap]
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]
    overlay = (1 - alpha) * original_image_0_1 + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)
