import cv2  # Computer vision library
import numpy as np  # Scientific computing library
import object_detection  # Custom object detection program
from tensorflow import keras  # Library for neural networks
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from datetime import datetime

# Важно: файл должен находиться в том же каталоге, где и наш код.
filename = 'C:\Users\Focus\Desktop\trafficPrediction\SecondPrototype\video.mp4'
file_size = (1920, 1080)  # Предполагается размер файла и формат 1920x1080 mp4
scale_ratio = 1  # Опция для масштабирования до доли исходного размера.

# Сохраним выходное видео со следующим названием и фпс
output_filename = '${0}video_annotated.mp4'.format(filename)
output_frames_per_second = 30.0

# Загрузим нейронную сеть SSD, обученную на наборе данных COCO.
model_ssd = object_detection.load_ssd_coco()

# Загрузим обученную нейронную сеть
model_traffic_lights_nn = keras.models.load_model("traffic.h5")


def main():
    # Загрузим видео
    cap = cv2.VideoCapture(filename)

    output_fps = cap.get(cv2.CAP_PROP_FPS)  # фпс с оригинального видео
    # Создадим объект VideoWriter, чтобы мы могли сохранить вывод видео.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    result = cv2.VideoWriter(output_filename,
                             fourcc,
                             output_fps,  # output_frames_per_second,
                             file_size)

    frames_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_number = 0
    # Обработаем видео
    while cap.isOpened():

        # Каждой итерацией берём по одному кадру
        success, frame = cap.read()

        # Пока есть кадр из видео, продолжаем
        if success:

            # Изменим размер кадра
            width = int(frame.shape[1] * scale_ratio)
            height = int(frame.shape[0] * scale_ratio)
            frame = cv2.resize(frame, (width, height))

            # Сохраним оригинальный кадр
            original_frame = frame.copy()

            output_frame = object_detection.perform_object_detection_video(
                model_ssd, frame, model_traffic_lights=model_traffic_lights_nn)

            # Записываем кадр в выходной видеофайл
            result.write(output_frame)

            frame_number += 1
            print('{0}: Frame processeed: {1} / {2}'.format(datetime.now(), frame_number, frames_count))

        # Если кадров больше не осталось, заканчиваем цикл
        else:
            print('No frames left')
            break

    # Остановим, когда видео закончится
    cap.release()

    # Реализуем видос
    result.release()

    # Закроем все окна
    cv2.destroyAllWindows()


main()
