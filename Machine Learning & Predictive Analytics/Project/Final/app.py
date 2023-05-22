import streamlit as st
from streamlit_webrtc import webrtc_streamer,RTCConfiguration,WebRtcMode
import av
import cv2
import numpy as np
import cvlib as cv
from PIL import Image
from keras.models import load_model
from timeit import time
import warnings
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import traceback
import logging
import queue
from pathlib import Path
from typing import List, NamedTuple
from sample_utils.download import download_file
from sample_utils.turn import get_ice_servers
import threading


warnings.filterwarnings('ignore')



HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
PROTOTXT_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.prototxt.txt"

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class Detection_class(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray 

st.title("Test")

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

@st.cache_resource  # type: ignore
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))

COLORS = generate_label_colors()

download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)

lock = threading.Lock()
predictions = {}

bmi_model = load_model("best_model_vgg_complex.h5",compile=False)
sentiment_model = load_model("sentiment_standard.h5",compile=False)

cache_key = "object_detection_dnn"
if cache_key in st.session_state:
    net,encoder,predictions = st.session_state[cache_key]
else:
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))
    encoder = gdet.create_box_encoder('model_data/mars-small128.pb', batch_size=1)
    
    predictions = {}
    st.session_state[cache_key] = [net,encoder,predictions]

score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()
# webrtc_streamer(key="Sample")

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    h = image.shape[0]
    w = image.shape[1]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def pre_process_image(face_crop,face_gray,size=100):
    w,h = face_crop.shape[1],face_crop.shape[0]
    new_image = np.zeros((size,size,3))
    new_gray_image = np.zeros((48,48))
    if w>h:
        image = image_resize(face_crop,width=size)
        image_gray = image_resize(face_gray,width=48)
        height_offset = int((new_image.shape[0] - image.shape[0])/2)
        new_image[height_offset:int(image.shape[0]+height_offset), :image.shape[1]] = image
    elif h>w:
        image = image_resize(face_crop,height=size)
        image_gray = image_resize(face_gray,height=48)
        width_offset = int((new_image.shape[1] - image.shape[1])/2)
        new_image[:image.shape[0], width_offset:int(image.shape[1] + width_offset)] = image
    else:
        image = image_resize(face_crop,width=size,height=size)
        image_gray = image_resize(face_gray,width=48,height=48)
        new_image[:image.shape[0], :image.shape[1]] = image
    new_gray_image[:image_gray.shape[0], :image_gray.shape[1]] = image_gray
    new_gray_image = np.expand_dims(new_gray_image,axis=2)
    face_crop = new_image.astype("float") / 255.0
    face_gray = new_gray_image.astype("float")/255.0
    return face_crop,face_gray

def detect_face(frame,bboxes,bmi_model,sentiment_model):
    faces= []
    face_grays = []
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_crop = []
    face_gray = []
    (startX, startY) = int(bboxes[0]), int(bboxes[1])
    (endX, endY) = int(bboxes[2]), int(bboxes[3])    
    startX = startX - 100
    startY = startY -100
    endX = endX + 100
    endY = endY + 100
    startX = 0 if startX<1 else startX
    startY = 0 if startY<1 else startY 
    face_crop = np.copy(frame[startY:endY,startX:endX])
    face_gray = np.copy(gray_frame[startY:endY,startX:endX])
    face_crop,face_gray = pre_process_image(face_crop,face_gray,size=200)
    faces.append(face_crop)
    face_grays.append(face_gray)
    faces = np.array(faces, dtype="float32")
    face_grays = np.array(face_grays, dtype="float32")
    
    return faces,face_grays

class VideoProcessor:
    def recv(self,frame):
        print("Started")
        
    # Detect faces in the frame using cvlib
        try:
            img = frame.to_ndarray(format="bgr24")
            image = Image.fromarray(img[...,::-1])
            blob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5
                )
            net.setInput(blob)
            output = net.forward()
            h, w = img.shape[:2]

            # Convert the output array into a structured form.
            output = output.squeeze()  # (1, 1, N, 7) -> (N, 7)
            
            output = output[output[:, 2] >= score_threshold]
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.3, None)
            tracker = Tracker(metric)
            tracking = True
           
            detections = [
                Detection_class(
                    class_id=int(detection[1]),
                    label=CLASSES[int(detection[1])],
                    score=float(detection[2]),
                    box=[int(t*z) for t,z in zip(detection[3:7],[w,h,w,h])],
                )
                for detection in output
            ]
            if tracking:
                # print(detections[0].box[3])
                detections_list = [Detection(detection.box, detection.score, detection.class_id, encoder(img,[detection.box])) for detection in detections]
            else:
                detections_list = [Detection_YOLO(detection.box, detection.score, detection.class_id,) for detection in detections]
        except:
            traceback.print_exc() 
            print("1")
        try:
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections_list])
            scores = np.array([d.confidence for d in detections_list])
            indices = preprocessing.non_max_suppression(boxes, 1.0, scores)
            detections_list = [detections_list[i] for i in indices]
            
            if tracking:
                # Call the tracker
                tracker.predict()
                tracker.update(detections_list)
                
                for track in tracker.tracks:
                    print(track.is_confirmed())
                    # if not track.is_confirmed() or track.time_since_update > 1:
                    #     continue
                    bbox = track.to_tlbr()
                    print(str(track.track_id))
                    print(predictions)
                    if str(track.track_id) not in predictions:
                        faces,face_grays = detect_face(img,bbox,bmi_model,sentiment_model)
                        bmi_pred = bmi_model.predict(faces,batch_size=32)
                        sentiment_pred = sentiment_model.predict(face_grays,batch_size=32)
                        class_map = {0: 'angry',1: 'disgust',2: 'fear',3: 'happy',4: 'neutral',5: 'sad',6: 'surprise'}
                        sentiment_pred = [(class_map[sub_emotion.argmax()],max(sub_emotion)) for sub_emotion in sentiment_pred]
                        predictions[str(track.track_id)] = {"BMI":int(bmi_pred),"Sentiment":{"Status":sentiment_pred[0][0],"Confidence":sentiment_pred[0][1]}}
                    bmi = predictions[str(track.track_id)]["BMI"]
                    sentiment_confidence = predictions[str(track.track_id)]["Sentiment"]["Confidence"]
                    sentiment_status = predictions[str(track.track_id)]["Sentiment"]["Status"]
                    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    cv2.putText(img, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])+40), 0,1.5e-3 * img.shape[0], (0, 255, 0), 1)
                    cv2.putText(img, "BMI: " + str(bmi), (int(bbox[0]), int(bbox[1])+20), 0,1.5e-3 * img.shape[0], (0, 255, 0), 1)
                    cv2.putText(img, "Sentiment: " + str(sentiment_status) +" " + str(sentiment_confidence) , (int(bbox[0]), int(bbox[1])), 0,1.5e-3 * img.shape[0], (0, 255, 0), 1)

            for det in detections_list:
                bbox = det.to_tlbr()
                score = "%.2f" % round(det.confidence * 100, 2) + "%"
        except Exception as e:
            traceback.print_exc()
            print("Error")
            print("Extra")
            for detection in detections:
                caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
                color = COLORS[detection.class_id]
                xmin, ymin, xmax, ymax = detection.box

                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(
                    img,
                    caption,
                    (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        
        
        return av.VideoFrame.from_ndarray(img,format="bgr24")
    
webrtc_streamer(key="edge",mode=WebRtcMode.SENDRECV,video_processor_factory=VideoProcessor,media_stream_constraints={"video": True, "audio": False},rtc_configuration=RTC_CONFIGURATION,async_processing=False)