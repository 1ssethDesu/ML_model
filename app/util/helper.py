import cv2
import numpy as np

# Image processing functions from provided code
def image_process(image, input_shape=(640, 640)):
    # Read image as RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get image shape
    img_h, img_w, _ = img.shape

    # Resize image
    img = cv2.resize(img, input_shape)

    # Normalize image
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, *img.shape)
    img = img.astype(np.float32) / 255.0

    return img, img_h, img_w

def filter_detection(output, thresh=0.25):
    A = []

    prediction = output[0]
    prediction = prediction.transpose()

    for detection in prediction:
        class_id = detection[4:].argmax()
        confidence_score = detection[4:].max()

        new_detection = np.append(detection[:4], [class_id, confidence_score])
        A.append(new_detection)

    A = np.array(A)

    # Filter out the detections with confidence > thresh
    considerable_detections = [detection for detection in A if detection[-1] > thresh]
    considerable_detections = np.array(considerable_detections)

    return considerable_detections

# Apply non-maximum suppression
def NMS(boxes, conf_scores, iou_thresh=0.55):
    if len(boxes) == 0:
        return [], []
        
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        if len(order) == 0:
            keep.append(A)
            keep_confidences.append(conf)
            break

        xx1 = np.take(x1, indices=order)
        yy1 = np.take(y1, indices=order)
        xx2 = np.take(x2, indices=order)
        yy2 = np.take(y2, indices=order)

        keep.append(A)
        keep_confidences.append(conf)

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2 - xx1, 0)
        h = np.maximum(yy2 - yy1, 0)

        intersection = w * h

        other_areas = np.take(areas, indices=order)
        union = areas[idx] + other_areas - intersection

        iou = intersection / union

        boleans = iou < iou_thresh

        order = order[boleans]

    return keep, keep_confidences

def rescale_back(results, img_w, img_h):
    if len(results) == 0:
        return [], []
        
    cx, cy, w, h, class_id, confidence = results[:, 0], results[:, 1], results[:, 2], results[:, 3], results[:, 4], results[:, -1]
    cx = cx / 640.0 * img_w
    cy = cy / 640.0 * img_h
    w = w / 640.0 * img_w
    h = h / 640.0 * img_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    keep, keep_confidences = NMS(boxes, confidence)
    
    return keep, keep_confidences
