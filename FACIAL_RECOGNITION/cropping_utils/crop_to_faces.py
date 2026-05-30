# THIS IS INPLACE. BACK UP DATA FIRST
target_folder = "folder_with_raw_images" 

import os
import random
import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from tqdm import tqdm

font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)

# Face detector setup
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
detector = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(base_options=base_options))

deleted = 0

def crop_face_with_padding(image_path, pad_top=0.0, pad_side=0.0, pad_bottom=0.0):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    
    if not result.detections:
        return None, None
    
    detection = result.detections[0]
    
    # Check if both eyes are visible (frontal face)
    keypoints = detection.keypoints
    if len(keypoints) < 2:
        return None, None
    
    left_eye = keypoints[0]
    right_eye = keypoints[1]
    if not (left_eye and right_eye):
        return None, None
    
    bbox = detection.bounding_box
    face_w, face_h = bbox.width, bbox.height
    
    x1 = int(bbox.origin_x - face_w * pad_side)
    y1 = int(bbox.origin_y - face_h * pad_top)
    x2 = int(bbox.origin_x + face_w + face_w * pad_side)
    y2 = int(bbox.origin_y + face_h + face_h * pad_bottom)
    
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    cropped = img[y1:y2, x1:x2]
    # Return bbox relative to cropped image
    bbox_in_crop = [[int(bbox.origin_x - x1), int(bbox.origin_y - y1)], 
                    [int(bbox.origin_x + bbox.width - x1), int(bbox.origin_y + bbox.height - y1)]]
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)), bbox_in_crop

for person_id in tqdm(os.listdir(target_folder)):
    person_path = os.path.join(target_folder, person_id)
    if not os.path.isdir(person_path):
        continue
    for img_name in os.listdir(person_path):
        if not img_name.endswith('.png'):
            continue
        img_path = os.path.join(person_path, img_name)
        cropped, _ = crop_face_with_padding(img_path)
        if cropped is None:
            os.remove(img_path)
            deleted += 1
        else:
            cropped.save(img_path, quality=95)

print(f"Done. Deleted {deleted} images.")