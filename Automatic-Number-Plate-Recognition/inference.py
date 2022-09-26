# from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
import numpy as np
import pytesseract
import cv2
import re
import tensorflow as tf
import json

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = './exported_graph/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './exported_graph/label_map.pbtxt'

img_path = "example/1.jpeg"
image_np = cv2.imread(img_path)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.compat.v2.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

################## Preprocessing for Number plate OCR ###################
# gray scale
def gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r"./preprocess/img_gray.png",img)
    return img

# blur
def blur(img) :
    img_blur = cv2.GaussianBlur(img,(5,5),0)
    cv2.imwrite(r"./preprocess/img_blur.png",img)    
    return img_blur

# threshold
def threshold(img):
    #pixels with value below 100 are turned black (0) and those with higher value are turned white (255)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]    
    cv2.imwrite(r"./preprocess/img_threshold.png",img)
    return img

################## ################################## ###################

def run_inference_for_single_image(image, graph, tensor_dict, sess):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


with detection_graph.as_default():
    with tf.compat.v1.Session() as sess:
        # Get handles to input and output tensors
        ops = tf.compat.v1.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(
            tensor_name)

        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np, detection_graph, tensor_dict, sess)

        text = ""
        max_boxes_to_draw = output_dict['detection_boxes'].shape[0]
        for i in range(min(max_boxes_to_draw, output_dict['detection_boxes'].shape[0])):
            if output_dict['detection_scores'][i] > 0.80:
                if output_dict['detection_classes'][i] in category_index.keys():
                    class_name = category_index[output_dict['detection_classes'][i]]['name']
                    ymin = output_dict['detection_boxes'][i][0]
                    xmin = output_dict['detection_boxes'][i][1]
                    ymax = output_dict['detection_boxes'][i][2]
                    xmax = output_dict['detection_boxes'][i][3]

                    im_height, im_width, im_channel = image_np.shape
                    (xminn, xmaxx, yminn, ymaxx) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                    crop_img=tf.image.crop_to_bounding_box(image_np,int(yminn), int(xminn), int(ymaxx-yminn), int(xmaxx-xminn))
                    crop_img=image_np[int(yminn):int(ymaxx),int(xminn):int(xmaxx)]

                    img_path = "./preprocess/" + img_path.split("/")[-1].split(".")[0] + "_cropped.png"
                    # Finding contours 
                    im_gray = gray(crop_img)
                    im_blur = blur(im_gray)
                    im_thresh = threshold(im_blur)

                    config = ('-l eng --oem 1 --psm 3')
                    # pytessercat
                    text = pytesseract.image_to_string(im_thresh, config=config)
                    # print text
                    text = re.sub("[^A-Z0-9 ]", "", text.strip())
                    text = re.sub("[^A-Z0-9 -]", "", text.strip().replace(" ", "-"))


json_response = json.dumps({
    "output": text
})
print(json_response)