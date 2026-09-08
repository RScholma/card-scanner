"""
Score `recognize_card_number.recognize` against the labelled number crops.

    python eval_recognizer.py            # val split
    python eval_recognizer.py --all      # train + val
    python eval_recognizer.py --show     # print per-image detail
"""

import argparse
import glob
import os
import time

import cv2

from recognize_card_number import recognize


def load_pairs(splits):
    pairs = []
    for split in splits:
        for label_path in sorted(glob.glob(f"dataset/card_number_crops/{split}/labels/*.txt")):
            name = os.path.splitext(os.path.basename(label_path))[0]
            image_path = f"dataset/card_number_crops/{split}/images/{name}.png"
            if os.path.exists(image_path):
                with open(label_path) as fh:
                    pairs.append((name, image_path, fh.read().strip().upper()))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="use train+val, not just val")
    parser.add_argument("--show", action="store_true", help="print every image")
    args = parser.parse_args()

    splits = ["train", "val"] if args.all else ["val"]
    pairs = load_pairs(splits)
    if not pairs:
        raise SystemExit("No labelled crops found.")

    exact = 0
    wrong = 0
    declined = 0
    scored = []          # (confidence, was_correct) for answers actually given
    started = time.time()

    for name, image_path, truth in pairs:
        result = recognize(cv2.imread(image_path))
        got = result.text or ""
        hit = got == truth
        exact += hit
        if not got:
            declined += 1
        else:
            wrong += not hit
            scored.append((result.confidence, hit))
        if args.show:
            mark = "OK " if hit else ("BAD" if got else "-- ")
            print(f"{mark} {name:<10} truth={truth:<12} got={got or '(none)':<12} "
                  f"conf={result.confidence:.2f}")

    n = len(pairs)
    elapsed = time.time() - started
    print("-" * 62)
    print(f"split={'+'.join(splits)}  n={n}")
    print(f"exact match  {exact}/{n} ({exact / n:.1%})   "
          f"wrong {wrong}   declined {declined}")
    print(f"{elapsed / n:.2f}s per crop")

    # Is `confidence` safe to threshold on? Only if precision climbs as the bar
    # rises - a wrong answer surviving a high threshold makes the score useless.
    print("\nconfidence threshold -> kept, of which correct")
    for bar in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7):
        kept = [hit for conf, hit in scored if conf >= bar]
        if kept:
            print(f"  >={bar:.1f}   kept {len(kept):>2}/{n}   "
                  f"precision {sum(kept) / len(kept):.0%}   "
                  f"recall {sum(kept) / n:.0%}")


if __name__ == "__main__":
    main()
