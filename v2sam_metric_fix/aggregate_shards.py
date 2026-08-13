"""Aggregate sharded raw per-pair results into the three reported aggregations."""
import json, glob, sys, os

out_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(out_dir, "raw_*.json")))
if not files:
    sys.exit(f"no raw_*.json in {out_dir}")

# shards preserve dataset order, so concatenating restores the global ordering
# that the stock truncation depends on
results = []
for f in files:
    results.extend(json.load(open(f)))

n_pairs = len(results)
flat_iou = [v for r in results for v in r["ious"]]
flat_dice = [v for r in results for v in r["dices"]]
n_obj = len(flat_iou)
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")

print(f"shards aggregated : {len(files)}")
print(f"image-pairs       : {n_pairs}")
print(f"object masks      : {n_obj}")
print(f"dropped by stock  : {n_obj-n_pairs} ({100*(n_obj-n_pairs)/n_obj:.1f}%)")
print("-" * 58)
print(f"stock      (truncated to size)  IoU {100*mean(flat_iou[:n_pairs]):6.2f}   Dice {100*mean(flat_dice[:n_pairs]):6.2f}")
print(f"per-object (all masks)          IoU {100*mean(flat_iou):6.2f}   Dice {100*mean(flat_dice):6.2f}")
print(f"per-pair   (pair then mean)     IoU {100*mean([mean(r['ious']) for r in results]):6.2f}   "
      f"Dice {100*mean([mean(r['dices']) for r in results]):6.2f}")
