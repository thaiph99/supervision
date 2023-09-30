# Note for calculating mAP

## MMRotate

### Function flow

```None
DOTAMetric  -->     eval_rbbox_map  |-->    tpfp_default        -->     box_iou_rotated
                                    |-->    average_precisions
```

### Techniques

- Use multithread
- Calculate rotated bounding box IoU by matrix multiplication
- Clean code

### Questions

- Why don't use `non_max_suppression` to ignore many duplication bboxes?
(Don't find nms in the calculating mAP progress)

## Yolov5 OBB

### Techniques

- Use horizontal bounding box to calculate IoU rotation
