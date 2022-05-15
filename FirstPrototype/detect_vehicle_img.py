import cv2  # Computer vision library
import numpy as np  # Scientific computing library
import vehicle_detection  # Custom object detection program
from tensorflow import keras  # Library for neural networks
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# Обнаружение объектов в пакете файлов изображений.
files = vehicle_detection.get_files('test_images/*.jpg')

# Загрузим нейронную сеть SSD, обученную на наборе данных COCO.
model_ssd = vehicle_detection.load_ssd_coco()

# Просмотр всех файлов изображений и определение объектов.
for file in files:
    (img, out, file_name) = vehicle_detection.perform_object_detection(
        model_ssd, file, save_annotated=True)
    print(file, out)
