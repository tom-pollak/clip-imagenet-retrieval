from typing import Iterable
from clip_index.text.query import AnnoyImage


class QueryStats:
    def __init__(self, tp, tn, fp, fn, n=None):
        if n is not None:
            assert tp + tn + fp + fn == n
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn

    def accuracy(self):
        """Not a good meaure because inbalanced dataset -- negatives vastly outweigh positives"""
        return (self.tn + self.tp) / (self.tp + self.fp + self.tn + self.fn)

    def precision(self):
        """Measures false positives"""
        if self.tp + self.fp == 0:
            return 1
        return self.tp / (self.tp + self.fp)

    def recall(self):
        """Measures false negatives"""
        return self.tp / (self.tp + self.fn)

    def f1(self):
        """Harmonic mean of precision & recall and is a better measure of accuracy"""
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0
        return 2 * (p * r) / (p + r)

    def format_stats(self) -> str:
        return "accuracy: %.4f, precision: %.4f, recall: %.4f, f1: %.4f" % (
            self.accuracy(),
            self.precision(),
            self.recall(),
            self.f1(),
        )


def annoy_query_stats(
    q: str, ann_imgs: Iterable[AnnoyImage], total_in_class: int, total_dataset: int
) -> QueryStats:
    tp = 0
    fp = 0
    #     incorrect_clss = []
    for im in ann_imgs:
        assert im.imagenet_classes is not None
        if q in im.imagenet_classes:
            tp += 1
        else:
            fp += 1
    #             incorrect_clss.append((im.image_path, im.imagenet_classes, im.dist))
    fn = total_in_class - tp
    tn = total_dataset - tp - fp - fn
    qstats = QueryStats(tp, tn, fp, fn, total_dataset)
    return qstats
