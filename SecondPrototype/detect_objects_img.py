import cv2  # Computer vision library
import numpy as np  # Scientific computing library
import object_detection  # Custom object detection program
from tensorflow import keras  # Library for neural networks
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# Загрузим модель Inception V3.
model_inception = InceptionV3(weights='imagenet', include_top=True, input_shape=(299, 299, 3))

# Обнаружение цвета светофора в пакете файлов изображений.
files = object_detection.get_files('test_images/*.jpg')

# Загрузим нейронную сеть SSD, обученную на наборе данных COCO.
model_ssd = object_detection.load_ssd_coco()

# Загрузим нашу обученную нейронную сеть.
model_traffic_lights_nn = keras.models.load_model("traffic.h5")

# Просмотр всех файлов изображений и определение цветов светофоров.
for file in files:
    # Будем сравнивать результаты, полученные нашей нейронной сетью и моделью Inception V3 с весами imagenet
    # (необязательно нам это делать; только время занимает)
    # Изменим размер изображения.
    img = cv2.resize(preprocess_input(cv2.imread(file)), (299, 299))  # Раскомментить для сравнения
    # Сгенерируем предикшены
    out_inception = model_inception.predict(np.array([img]))  # Раскомментить для сравнения
    # Декодим предикшены
    out_inception = imagenet_utils.decode_predictions(out_inception)  # Раскомментить для сравнения
    obj = 0  # Раскомментить для сравнения
    while obj < len(out_inception[0]):  # Раскомментить для сравнения
        print("Prediction for ", file, ':', out_inception[0][obj][1], out_inception[0][obj][2], "%")  # Раскомментить
        obj += 1  # Раскомментить для сравнения
    # Выведим сводные данные модели Inception V3
    # model_inception.summary() # Раскомментить для сравнения

    # Для каждого файла из папки запускаем процесс детекта объектов
    (img_out, out, file_name) = object_detection.perform_object_detection(
        model_ssd, file, save_annotated=True, model_traffic_lights=model_traffic_lights_nn)
    # print(file, out)
