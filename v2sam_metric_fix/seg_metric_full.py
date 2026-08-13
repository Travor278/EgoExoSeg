import json, os
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmengine.logging import MMLogger, print_log


@METRICS.register_module()
class SegMetricFull(BaseMetric):
    """Truncation-free companion to :class:`SegMetric`.

    ``SegMetric.process`` appends one entry per *object mask*, but
    ``BaseMetric.evaluate(size)`` is handed ``size = len(dataset)`` -- the number
    of image-pairs.  ``collect_results_cpu`` then does ``result_part[:size]``
    (dist.py L37-38 for world_size==1, L84 for the gathered path), so whenever a
    pair carries more than one object the tail of the list is silently dropped.

    Appending one entry per *pair* instead makes ``len(self.results) == size``
    exactly, so the slice becomes a no-op and every mask survives.  From that we
    report three aggregations:

      stock_IoU     - flat per-object list truncated to ``size``; reproduces what
                      the unmodified SegMetric yields on a single GPU
      perobject_IoU - mean over every object mask (no truncation)
      perpair_IoU   - mean within a pair first, then mean over pairs

    ``out_file`` dumps the raw per-pair lists so a sharded run can be aggregated
    afterwards without re-running inference.
    """

    def __init__(self, iou_metrics=None, out_file=None, **kwargs):
        super().__init__(**kwargs)
        self.iou_metrics = iou_metrics or ["IoU", "Dice"]
        self.out_file = out_file
        self.results = []

    def process(self, data_batch, data_samples):
        gt_masks_all = data_batch["data"].get("masks", None)
        for i in range(len(gt_masks_all)):
            gt_masks = gt_masks_all[i]
            pred_masks = data_samples[i]["pred_masks"]
            ious, dices = [], []
            for j in range(gt_masks.shape[0]):
                pred = pred_masks[j].float()
                gt = gt_masks[j].float().to(pred.device)
                inter = (pred * gt).sum()
                union = (pred + gt - pred * gt).sum()
                ious.append((inter / (union + 1e-6)).item())
                dices.append(((2 * inter) / (pred.sum() + gt.sum() + 1e-6)).item())
            # one entry per image-pair -> len(results) == size -> no truncation
            self.results.append(dict(ious=ious, dices=dices))

    def compute_metrics(self, results):
        logger = MMLogger.get_current_instance()
        if not results:
            print_log("No results collected in SegMetricFull!", logger=logger)
            return {}

        if self.out_file:
            os.makedirs(os.path.dirname(self.out_file) or ".", exist_ok=True)
            with open(self.out_file, "w") as f:
                json.dump(results, f)
            print_log(f"raw per-pair results -> {self.out_file}", logger=logger)

        n_pairs = len(results)
        flat_iou = [v for r in results for v in r["ious"]]
        flat_dice = [v for r in results for v in r["dices"]]
        n_obj = len(flat_iou)
        mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")

        perobject_iou, perobject_dice = mean(flat_iou), mean(flat_dice)
        perpair_iou = mean([mean(r["ious"]) for r in results])
        perpair_dice = mean([mean(r["dices"]) for r in results])
        stock_iou, stock_dice = mean(flat_iou[:n_pairs]), mean(flat_dice[:n_pairs])

        print_log("=" * 60, logger=logger)
        print_log(f"image-pairs (size) : {n_pairs}", logger=logger)
        print_log(f"object masks total : {n_obj}", logger=logger)
        print_log(f"masks dropped by stock truncation: {max(0, n_obj - n_pairs)}"
                  f"  ({100.0 * max(0, n_obj - n_pairs) / max(n_obj, 1):.1f}%)", logger=logger)
        print_log("-" * 60, logger=logger)
        print_log(f"stock     (truncated to size) IoU {100*stock_iou:6.2f}  Dice {100*stock_dice:6.2f}", logger=logger)
        print_log(f"per-object (all masks)        IoU {100*perobject_iou:6.2f}  Dice {100*perobject_dice:6.2f}", logger=logger)
        print_log(f"per-pair   (pair then mean)   IoU {100*perpair_iou:6.2f}  Dice {100*perpair_dice:6.2f}", logger=logger)
        print_log("=" * 60, logger=logger)

        return dict(n_pairs=n_pairs, n_objects=n_obj,
                    stock_IoU=stock_iou, stock_Dice=stock_dice,
                    perobject_IoU=perobject_iou, perobject_Dice=perobject_dice,
                    perpair_IoU=perpair_iou, perpair_Dice=perpair_dice)
