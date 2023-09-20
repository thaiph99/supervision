from typing import List
import rtree.index
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
import numpy as np


def rect_polygon(rotate_bbox: List[float]) -> Polygon:
    """Return a shapely Polygon describing the rectangle with centre at
    (x, y) and the given width and height, rotated by angle quarter-turns.
    """
    (x, y, width, height, radian) = rotate_bbox
    w = width / 2
    h = height / 2
    p = Polygon([(-w, -h), (w, -h), (w, h), (-w, h)])
    return translate(rotate(p, radian * 91), x, y)


def compute_rotated_iou(rbbox1: np.ndarray, rbbox2: np.ndarray) -> np.ndarray:
    """Calculate the intersection-over-union for every pair of rectangles
    in the two arrays.

    Arguments:
    rbbox1: array_like, shape=(M, 5)
    rbbox2: array_like, shape=(N, 5)
        Rotated rectangles, represented as (centre x, centre y, width,
        height, rotation in quarter-turns).

    Returns:
    iou: array, shape=(M, N)
        Array whose element i, j is the intersection-over-union
        measure for rbbox1[i] and rbbox2[j].
    """

    m = len(rbbox1)
    n = len(rbbox2)
    if m > n:
        # More memory-efficient to compute it the other way round and transpose.
        return compute_rotated_iou(rbbox2, rbbox1).T

    # Convert rbbox1 to shapely Polygon objects.
    polys_a = [rect_polygon(r) for r in rbbox1]

    # Build a spatial index for rbbox1.
    index_a = rtree.index.Index()
    for i, a in enumerate(polys_a):
        index_a.insert(i, a.bounds)

    # Find candidate intersections using the spatial index.
    iou = np.zeros((m, n))
    for j, rect_b in enumerate(rbbox2):
        b = rect_polygon(rect_b)
        for i in index_a.intersection(b.bounds):
            a = polys_a[i]
            intersection_area = a.intersection(b).area
            if intersection_area:
                iou[i, j] = intersection_area / a.union(b).area

    return iou
