"""Standalone exact-duplicate audit of the Kermany dataset.
Detects same-class copies and files whose bytes appear under BOTH classes
(label conflict = unusable). --quarantine moves offending files to
data/quarantine/ (conflicts: all copies; same-class dups: all but one).
Run: python dedup_audit.py [--root data/chest_xray/train] [--quarantine]"""
import argparse
import hashlib
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/chest_xray/train")
    ap.add_argument("--quarantine", action="store_true")
    args = ap.parse_args()
    groups = {}
    for cls in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, cls)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            p = os.path.join(d, fname)
            with open(p, "rb") as f:
                groups.setdefault(hashlib.md5(f.read()).hexdigest(), []).append(p)
    dup_same = conflicts = 0
    for paths in groups.values():
        classes = {os.path.basename(os.path.dirname(p)) for p in paths}
        if len(classes) > 1:
            conflicts += len(paths)
            print("CONFLICT (both classes):", *paths, sep="\n  ")
            to_move = paths
        elif len(paths) > 1:
            dup_same += len(paths) - 1
            print("DUP:", *paths, sep="\n  ")
            to_move = paths[1:]
        else:
            continue
        if args.quarantine:
            for p in to_move:
                dest = os.path.join("data", "quarantine", os.path.relpath(p, args.root))
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(p, dest)
    print(f"\n{dup_same} same-class duplicate file(s); {conflicts} cross-class conflict file(s).")


if __name__ == "__main__":
    main()