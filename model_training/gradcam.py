"""
Grad-CAM (Selvaraju et al., 2017): visualizes which regions of an X-ray the
model actually weighted most heavily for its prediction. Standard practice
in current chest X-ray classification literature, and directly useful here
given some sample images in this dataset carry corner markers/artifacts —
Grad-CAM lets you show the model is attending to lung fields, not artifacts,
rather than just claiming it.
"""

import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras


def find_backbone_submodel(model):
    """The backbone is embedded as a nested Functional model within the
    outer model. Its exact layer name (e.g. "densenet121") isn't reliable
    across Keras versions/backbones, so find it by type instead — the first
    layer that is itself a keras.Model."""
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("No nested backbone sub-model found in this model.")


def find_last_conv_layer(backbone):
    """Last layer with a 4D (batch, H, W, C) output — i.e. the last spatial
    feature map before pooling. Found by shape rather than a hardcoded name
    since exact layer names can shift across Keras/TF versions."""
    for layer in reversed(backbone.layers):
        if len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No 4D (conv-like) layer found in backbone.")


def make_gradcam_heatmap(model, image_batch, class_index, last_conv_layer_name=None):
    """
    model: the full pneumonia model (Input -> backbone -> GAP -> Dropout -> Dense)
    image_batch: a single preprocessed image, shape (1, H, W, 3)
    class_index: which output neuron to explain (0=NORMAL, 1=PNEUMONIA)
    Returns a (H, W) heatmap in [0, 1], same spatial size as the last conv layer.
    """
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
        raise RuntimeError(
            "Gradient is None — check that last_conv_layer_name is actually "
            "upstream of the output in the graph."
        )

    # Global-average-pool the gradients over space -> per-channel "importance" weight
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + keras.backend.epsilon())
    return heatmap.numpy()


def overlay_heatmap(original_image_0_1, heatmap, alpha=0.4, colormap="jet"):
    """original_image_0_1: (H, W, 3) in [0, 1]. Returns an (H, W, 3) uint8
    image with the heatmap overlaid, resized to match."""
    import matplotlib.cm as cm

    h, w = original_image_0_1.shape[:2]
    heatmap_resized = tf.image.resize(heatmap[..., np.newaxis], (h, w)).numpy().squeeze()

    cmap = matplotlib.colormaps.get_cmap(colormap) if hasattr(matplotlib.colormaps, 'get_cmap') else matplotlib.colormaps[colormap]
    colored_heatmap = cmap(heatmap_resized)[:, :, :3]  # drop alpha channel

    overlay = (1 - alpha) * original_image_0_1 + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)
