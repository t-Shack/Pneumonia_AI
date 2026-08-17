"""Best-model selection — v3. Composite now ranks on BALANCED accuracy
(mean per-class recall) for primary and external components, so a model that
flags everything as pneumonia can no longer win on raw accuracy alone."""
import json
import os
import config


def load_json_if_exists(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _score(result):
    return result.get("balanced_accuracy", result["test_accuracy"])


def external_score_is_trustworthy(r):
    report = r.get("classification_report", {})
    return all(report.get(c, {}).get("support", 0) > 0 for c in config.CLASS_NAMES)


def mean_robustness_accuracy(r):
    accs = [lv["accuracy"] for deg in r["degradations"].values() for lv in deg.values()]
    return sum(accs) / len(accs) if accs else None


def select_by_accuracy(test_eval):
    scored = sorted(((l, _score(r), r["roc_auc"]) for l, r in test_eval.items()),
                    key=lambda t: (t[1], t[2]), reverse=True)
    winner, acc, auc = scored[0]
    return winner, {"strategy": "accuracy",
                    "ranking": [{"label": l, "balanced_accuracy": a, "roc_auc": r} for l, a, r in scored]}


def select_by_composite(test_eval, external_eval, robustness_eval):
    weights = dict(config.COMPOSITE_WEIGHTS)
    scores, details = {}, {}
    for label, result in test_eval.items():
        components = {"primary": _score(result)}
        used = {"primary": weights["primary"]}
        if external_eval and label in external_eval:
            ext = external_eval[label]
            if external_score_is_trustworthy(ext):
                components["external"] = _score(ext)
                used["external"] = weights["external"]
            else:
                details.setdefault("warnings", []).append(
                    f"{label}: external eval has a zero-support class - excluded.")
        if robustness_eval and label in robustness_eval:
            m = mean_robustness_accuracy(robustness_eval[label])
            if m is not None:
                components["robustness"] = m
                used["robustness"] = weights["robustness"]
        total = sum(used.values())
        norm = {k: v / total for k, v in used.items()}
        scores[label] = sum(components[k] * norm[k] for k in components)
        details[label] = {"components": components, "weights_used": norm,
                          "composite_score": scores[label]}
    return max(scores, key=scores.get), {"strategy": "composite", "details": details}


def main():
    test_eval = load_json_if_exists(os.path.join(config.METRICS_DIR, "test_evaluation.json"))
    if not test_eval:
        print("test_evaluation.json not found - run evaluate.py first.")
        return
    if config.BEST_MODEL_STRATEGY == "composite":
        ext = load_json_if_exists(os.path.join(config.METRICS_DIR, "external_evaluation.json"))
        rob = load_json_if_exists(os.path.join(config.METRICS_DIR, "robustness_evaluation.json"))
        if not ext and not rob:
            print("Composite requested but no external/robustness results - falling back to accuracy.")
            winner, info = select_by_accuracy(test_eval)
        else:
            winner, info = select_by_composite(test_eval, ext, rob)
    else:
        winner, info = select_by_accuracy(test_eval)
    print(f"\nBest model selected: {winner.upper()} (strategy: {info['strategy']})")
    print(json.dumps(info, indent=2))
    dep_path = os.path.join(config.OUTPUT_DIR, "deployment_config.json")
    if not os.path.exists(dep_path):
        print("deployment_config.json not found - run train.py first.")
        return
    with open(dep_path) as f:
        dep = json.load(f)
    dep["best_model"] = winner
    with open(dep_path, "w") as f:
        json.dump(dep, f, indent=2)
    with open(os.path.join(config.METRICS_DIR, "best_model_selection.json"), "w") as f:
        json.dump({"winner": winner, **info}, f, indent=2)
    print(f'Wrote "best_model": "{winner}" to deployment_config.json')


if __name__ == "__main__":
    main()