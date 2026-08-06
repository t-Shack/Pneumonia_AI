"""
Picks the best-performing model and records it in deployment_config.json as
"best_model" — that's what the webapp reads to know which single .keras
file to load and serve. Run this after evaluate.py (required), and
optionally after evaluate_external.py / robustness_test.py if you want the
"composite" strategy.

Two strategies (config.BEST_MODEL_STRATEGY):
  "accuracy"  — primary test accuracy, ROC-AUC as tiebreaker. Simple,
                always available as soon as evaluate.py has run.
  "composite" — weighted blend of primary accuracy, external accuracy, and
                mean robustness accuracy (config.COMPOSITE_WEIGHTS). Falls
                back to whatever's actually available, renormalizing
                weights over just those components, with a clear warning —
                it does NOT silently pretend to be more rigorous than the
                data on hand actually supports.

Sanity check built in: if a model's external evaluation has zero support
for either class (i.e. the external set is missing a class entirely — a
real issue this exact project ran into), that model's external score is
excluded from the composite rather than trusted at face value.

Run:
    python select_best_model.py
"""

import json
import os

import config


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def external_score_is_trustworthy(model_external_result):
    """False if either class had zero support in the external set — see
    evaluate_external.py's warning for the same check at evaluation time."""
    report = model_external_result.get("classification_report", {})
    for class_name in config.CLASS_NAMES:
        if report.get(class_name, {}).get("support", 0) == 0:
            return False
    return True


def mean_robustness_accuracy(model_robustness_result):
    accs = []
    for degradation in model_robustness_result["degradations"].values():
        for level in degradation.values():
            accs.append(level["accuracy"])
    return sum(accs) / len(accs) if accs else None


def select_by_accuracy(test_eval):
    scored = [
        (label, result["test_accuracy"], result["roc_auc"])
        for label, result in test_eval.items()
    ]
    scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
    winner, acc, auc = scored[0]
    return winner, {
        "strategy": "accuracy",
        "ranking": [{"label": l, "test_accuracy": a, "roc_auc": r} for l, a, r in scored],
    }


def select_by_composite(test_eval, external_eval, robustness_eval):
    weights = dict(config.COMPOSITE_WEIGHTS)
    scores = {}
    details = {}

    for label, result in test_eval.items():
        components = {"primary": result["test_accuracy"]}
        used_weights = {"primary": weights["primary"]}

        if external_eval and label in external_eval:
            ext_result = external_eval[label]
            if external_score_is_trustworthy(ext_result):
                components["external"] = ext_result["test_accuracy"]
                used_weights["external"] = weights["external"]
            else:
                details.setdefault("warnings", []).append(
                    f"{label}: external evaluation has a zero-support class — excluded from composite score."
                )

        if robustness_eval and label in robustness_eval:
            mean_acc = mean_robustness_accuracy(robustness_eval[label])
            if mean_acc is not None:
                components["robustness"] = mean_acc
                used_weights["robustness"] = weights["robustness"]

        weight_sum = sum(used_weights.values())
        normalized_weights = {k: v / weight_sum for k, v in used_weights.items()}
        composite = sum(components[k] * normalized_weights[k] for k in components)

        scores[label] = composite
        details[label] = {"components": components, "weights_used": normalized_weights, "composite_score": composite}

    winner = max(scores, key=scores.get)
    return winner, {"strategy": "composite", "details": details}


def main():
    eval_path = os.path.join(config.METRICS_DIR, "test_evaluation.json")
    test_eval = load_json_if_exists(eval_path)
    if not test_eval:
        print(f"{eval_path} not found — run evaluate.py first.")
        return

    strategy = config.BEST_MODEL_STRATEGY
    if strategy == "composite":
        external_eval = load_json_if_exists(os.path.join(config.METRICS_DIR, "external_evaluation.json"))
        robustness_eval = load_json_if_exists(os.path.join(config.METRICS_DIR, "robustness_evaluation.json"))
        if not external_eval and not robustness_eval:
            print("BEST_MODEL_STRATEGY is 'composite' but neither external_evaluation.json nor "
                  "robustness_evaluation.json exist yet — falling back to 'accuracy' for this run.")
            winner, selection_info = select_by_accuracy(test_eval)
        else:
            winner, selection_info = select_by_composite(test_eval, external_eval, robustness_eval)
    else:
        winner, selection_info = select_by_accuracy(test_eval)

    print(f"\nBest model selected: {winner.upper()}  (strategy: {selection_info['strategy']})")
    print(json.dumps(selection_info, indent=2))

    deployment_config_path = os.path.join(config.OUTPUT_DIR, "deployment_config.json")
    if not os.path.exists(deployment_config_path):
        print(f"{deployment_config_path} not found — run train.py at least once first "
              "(it's what creates this file).")
        return

    with open(deployment_config_path) as f:
        deployment_config = json.load(f)
    deployment_config["best_model"] = winner
    with open(deployment_config_path, "w") as f:
        json.dump(deployment_config, f, indent=2)
    print(f"\nWrote \"best_model\": \"{winner}\" to {deployment_config_path}")

    selection_out_path = os.path.join(config.METRICS_DIR, "best_model_selection.json")
    with open(selection_out_path, "w") as f:
        json.dump({"winner": winner, **selection_info}, f, indent=2)
    print(f"Full selection reasoning saved to {selection_out_path} (for your methodology section).")


if __name__ == "__main__":
    main()
