"""
Logit/probability diagnostic: prints true label, raw logits and final
probabilities per true class, on the internal test set and the external
NIH set, at threshold 0.5 and at the deployed decision threshold.
Run after train.py:
python logit_diagnostic.py
"""
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import config
import losses  # noqa: F401 — needed so load_model can deserialize the focal loss
from data_pipeline import get_generators, get_external_test_generator
from evaluate import discover_models

PNEU_IDX = config.CLASS_NAMES.index("PNEUMONIA")
NORM_IDX = config.CLASS_NAMES.index("NORMAL")


def collect_images(gen):
    gen.reset()
    xs, ys = [], []
    for _ in range(len(gen)):
        x, y = next(gen)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.argmax(np.concatenate(ys), axis=1)


def make_logit_model(model):
    """Same network, but with the head activation removed -> raw logits."""
    out = model.get_layer("output")
    linear = tf.keras.layers.Dense(out.units, activation="linear", name="output_logits")
    logits = linear(model.get_layer("head_dropout").output)  # builds the layer
    linear.set_weights(out.get_weights())
    return tf.keras.Model(inputs=model.inputs, outputs=logits)


def report(label, model, images, y_true, set_name, thresholds):
    logit_model = make_logit_model(model)
    probs = model.predict(images, verbose=0)
    logits = logit_model.predict(images, verbose=0)
    p = probs[:, PNEU_IDX]
    print(f"\n=== {label.upper()} on {set_name} ===")
    for idx, name in enumerate(config.CLASS_NAMES):
        m = y_true == idx
        if m.sum() == 0:
            print(f"  true {name}: none present")
            continue
        pp = p[m]
        line = (f"  true {name:9s} n={m.sum():4d} | P(PNEUM): "
                f"mean={pp.mean():.3f} min={pp.min():.3f} max={pp.max():.3f}")
        for t in thresholds:
            line += f" | called PNEUM @{t:.3f}: {(pp > t).mean():6.1%}"
        print(line)
    nm = np.where(y_true == NORM_IDX)[0]
    if len(nm):
        print("  first 10 true NORMAL examples (logits -> probs):")
        for i in nm[:10]:
            print(f"    logit[NORM]={logits[i, NORM_IDX]:+.3f} "
                  f"logit[PNEUM]={logits[i, PNEU_IDX]:+.3f} -> "
                  f"probs={np.round(probs[i], 3).tolist()}")


def main():
    thresholds = [0.5]
    dep_path = os.path.join(config.OUTPUT_DIR, "deployment_config.json")
    if os.path.exists(dep_path):
        with open(dep_path) as f:
            thresholds.append(json.load(f)["decision_threshold"])
    print(f"Thresholds examined: {thresholds}")

    _, _, test_gen = get_generators()
    x_test, y_test = collect_images(test_gen)

    x_ext = y_ext = None
    if os.path.isdir(config.EXTERNAL_TEST_DIR):
        x_ext, y_ext = collect_images(get_external_test_generator())
    else:
        print("External NIH dir not found; skipping external set.")

    for label, path in discover_models().items():
        model = load_model(path)
        report(label, model, x_test, y_test, "INTERNAL TEST", thresholds)
        if x_ext is not None:
            report(label, model, x_ext, y_ext, "EXTERNAL NIH", thresholds)


if __name__ == "__main__":
    main()