import os
import time
import cvut
import torch
import argparse
import numpy as np
import cv2

from trackers import BYTETracker

_FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_DICT = dict()
for i in range(256):
    _color_random = np.random.randint(0, 256, (3,), dtype='uint8')
    COLOR_DICT[i] = tuple(_color_random.tolist())
COLOR_LEN = len(COLOR_DICT)

def draw_track(image, bboxes, ids, labels=None, classnames=None,
                thickness=1,color=None, font=_FONT, font_size=0.5, font_thickness=1):
    """
    image (np.uint8) shape [H,W,3], RGB image
    bboxes (np.int/np.float/list) shape [N,4], format [x1, y1, x2, y2]
    ids (np.int/np.float/list) shape [N]
    labels (np.int/list) shape [N,], start-from-0. None is not used.
    classnames (list) of string, len [N,]. None is not used.
    """
    image_ = image.copy()

    if bboxes is not None:
        if labels is None:
            for bbox, track_id in zip(bboxes, ids):
                track_id = int(track_id)
                x1, y1, x2, y2 = [int(ele) for ele in bbox]
                _color = COLOR_DICT[track_id % len(COLOR_DICT)] if color is None else color
                cv2.rectangle(image_, (x1, y1), (x2, y2),
                              _color, thickness=thickness)
                cv2.putText(image_, "ID{}".format(track_id),
                            (int((x1+x2)/2), int((y1+y2)/2)),
                            font, font_size, _color, thickness=font_thickness)
        else:
            for bbox, track_id, label in zip(bboxes, ids, labels):
                label = int(label)
                track_id = int(track_id)
                x1, y1, x2, y2 = [int(ele) for ele in bbox]
                _color = COLOR_DICT[track_id % len(COLOR_DICT)] if color is None else color
                cv2.rectangle(image_, (x1, y1), (x2, y2),
                              _color, thickness=thickness)
                text = "cls{}-ID{}".format(label, track_id) if classnames is None \
                    else "{}-ID{}".format(classnames[label], track_id)
                cv2.putText(
                    image_, text, (int((x1+x2)/2), int((y1+y2)/2)),
                    font, font_size, _color, thickness=font_thickness)

    return image_

def inference_tracker(tracker, result):
    det_bboxes = np.concatenate(result)
    if det_bboxes.shape[-1] == 6:
        # det_bboxes contains mask ratio
        track_ids, det_bboxes, mask_ratios = tracker(det_bboxes)
        return track_ids, det_bboxes, mask_ratios
    else:
        track_ids, det_bboxes = tracker(det_bboxes)
        return track_ids, det_bboxes

def draw_track_result(track_ids, det_bboxes, det_labels,
                      img_or_path, classnames, thickness=4,
                      font_size=1.0, font_thickness=2,):
    
    num_objs = len(det_bboxes)
    if det_bboxes.shape[1] == 5:
        det_bboxes = det_bboxes[:, :4]

    # draw img
    # ori_img = mmcv.imread(img_or_path)
    if isinstance(img_or_path, str):
        ori_img = cv2.imread(img_or_path)
    else: 
        ori_img = img_or_path
    img = draw_track(ori_img, det_bboxes, track_ids, det_labels,
                          classnames=classnames, thickness=thickness,
                          font_size=font_size, font_thickness=font_thickness)
    
    return img, ori_img, num_objs

def draw_mask_ratio_result(img, ratios, bboxes):
    _font = cv2.FONT_HERSHEY_SIMPLEX
    _font_size = 1
    _font_thickness = 2
    _color = (255, 255, 255)
    for (bbox, ratio) in zip(bboxes, ratios):
        x1, y1, _, _ = [int(ele) for ele in bbox]
        ratio = np.round(ratio, 2)
        img = cv2.putText(img, str(ratio), (x1+5, y1+30), _font,
                          _font_size, _color, thickness=_font_thickness)
    return img

def tracking(image:np.ndarray, tracker:BYTETracker, results:np.ndarray, det_labels: list(int), classes: list(str)):
    assert len(det_labels) == len(classes),"Size of det_labels and classes must equal"
    # track
    track_ids, det_bboxes = inference_tracker(tracker, results)
    det_labels = None
    img_tracked, _, num_objs = draw_track_result(track_ids, det_bboxes, det_labels,
                                            image, classes)
    
    return img_tracked

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # build tracker
    det_thr=0.4
    match_thr=0.9
    track_buffer=30
    fps=30
    tracker = BYTETracker(track_thresh=det_thr,
                          match_thresh=match_thr,
                          track_buffer=track_buffer,
                          fuse_det_score=False,
                          frame_rate=fps)
    with torch.no_grad():
        tracking()
