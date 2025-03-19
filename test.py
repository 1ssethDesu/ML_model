import onnxruntime
import cv2
import numpy as np


def image_process(image, input_shape=(640, 640)):

    #read image as RGBs
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    #get image shape
    img_h, img_w, _ = img.shape

    #resize image
    img = cv2.resize(img, input_shape)

    #normalize image
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, *img.shape)
    img = img = img.astype(np.float32) / 255.0

    return img, img_h, img_w

def filter_detection(output, thresh=0.25):
    A = []

    prediction = output[0]
    prediction = prediction.transpose()

    for detection in prediction:

        class_id = detection[4:].argmax()
        confidence_score = detection[4:].max()

        new_detection = np.append(detection[:4],[class_id,confidence_score])

        A.append(new_detection)

    A = np.array(A)

    # filter out the detections with confidence > thresh
    considerable_detections = [detection for detection in A if detection[-1] > thresh]
    considerable_detections = np.array(considerable_detections)

    return considerable_detections

# apply non-maximum suppression
def NMS(boxes, conf_scores, iou_thresh = 0.55):

    #  boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        xx1 = np.take(x1, indices= order)
        yy1 = np.take(y1, indices= order)
        xx2 = np.take(x2, indices= order)
        yy2 = np.take(y2, indices= order)

        keep.append(A)
        keep_confidences.append(conf)

        # iou = inter/union

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2-xx1, 0)
        h = np.maximum(yy2-yy1, 0)

        intersection = w*h

        # union = areaA + other_areas - intesection
        other_areas = np.take(areas, indices= order)
        union = areas[idx] + other_areas - intersection

        iou = intersection/union

        boleans = iou < iou_thresh

        order = order[boleans]

        # order = [2,0,1]  boleans = [True, False, True]
        # order = [2,1]

    return keep, keep_confidences

def rescale_back(results ,img_w, img_h):
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
    cx = cx/640.0 * img_w
    cy = cy/640.0 * img_h
    w = w/640.0 * img_w
    h = h/640.0 * img_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    keep, keep_confidences = NMS(boxes,confidence)
    print(np.array(keep).shape)
    return keep, keep_confidences


if __name__ == "__main__":

    classes = ['gum-disease', 'tooth-decay', 'tooth-loss']

    session = onnxruntime.InferenceSession("app/models/model.onnx")

    image = cv2.imread("test/c-54.jpg")
    img, img_h, img_w = image_process(image)
    # print(img_h, img_w, img.shape)

    #run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img})

    results = filter_detection(output)
    # print(results.shape)

    rescaled_results, confidences = rescale_back(results, img_w, img_h)

for res, conf in zip(rescaled_results, confidences):
    x1, y1, x2, y2, cls_id = res
    cls_id = int(cls_id)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    conf = "{:.2f}".format(conf)
    
    # Define the label in two lines
    class_label = f"{classes[cls_id]}"
    conf_label = f"{conf}"
    
    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box
    
    # Draw the class label (first line)
    cv2.putText(image, class_label, (x1, y1 - 24),  # Position above the bounding box
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text
    
    # Draw the confidence label (second line)
    cv2.putText(image, conf_label, (x1, y1 - 4),  # Position below the class label
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text

    # Display the result
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()