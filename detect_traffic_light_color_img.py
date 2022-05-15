import cv2  # Computer vision library
import numpy as np  # Scientific computing library
import object_detection  # Custom object detection program
from tensorflow import keras  # Library for neural networks
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

FILENAME = "test_red.jpg"

# Загрузим модель Inception V3.
model_inception = InceptionV3(weights='imagenet', include_top=True, input_shape=(299, 299, 3))

# Изменим размер изображения.
img = cv2.resize(preprocess_input(cv2.imread(FILENAME)), (299, 299))

# Сгенерируем предикшены
out_inception = model_inception.predict(np.array([img]))

# Декодим предикшены
out_inception = imagenet_utils.decode_predictions(out_inception)

print("Prediction for ", FILENAME, ": ", out_inception[0][0][1], out_inception[0][0][2], "%")

# Выведим сводные данные модели
model_inception.summary()

# Обнаружение цвета светофора в пакете файлов изображений.
files = object_detection.get_files('test_images/*.jpg')

# Загрузим нейронную сеть SSD, обученную на наборе данных COCO.
model_ssd = object_detection.load_ssd_coco()

# Загрузим нашу обученную нейронную сеть.
model_traffic_lights_nn = keras.models.load_model("traffic.h5")

# Просмотр всех файлов изображений и определение цветов светофоров.
for file in files:
    (img, out, file_name) = object_detection.perform_object_detection(
        model_ssd, file, save_annotated=True, model_traffic_lights=model_traffic_lights_nn)
    print(file, out)
