import tensorflow as tf  # Machine learning library
from tensorflow import keras  # Library for neural networks
import numpy as np  # Scientific computing library
import cv2  # Computer vision library
import glob  # Filename handling library

# Inception V3 model for Keras
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Для обнаружения объектов мы будем использовать предварительно обученную нейронную сеть,
# обученную на наборе данных COCO. Подробнее об этом наборе данных можно прочитать здесь:
#   https://content.alegion.com/datasets/coco-ms-coco-dataset
# COCO labels здесь: https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_label_map.pbtxt
LABEL_CAR = 3
LABEL_BUS = 6
LABEL_TRUCK = 8
LABEL_TRAFFIC_LIGHT = 10


def accept_box(boxes, box_index, tolerance):
    """
    Удалим повторяющиеся ограничивающие рамки.
    """
    box = boxes[box_index]

    for idx in range(box_index):
        other_box = boxes[idx]
        if abs(center(other_box, "x") - center(box, "x")) < tolerance and abs(
                center(other_box, "y") - center(box, "y")) < tolerance:
            return False

    return True


def get_files(pattern):
    """
    Создать список всех изображений в каталоге.

    :param:pattern str - паттерн имен файлов
    :return: Список файлов, соответствующих указанному паттерну.
    """
    files = []

    # Для каждого файла, соответствующего указанному шаблону
    for file_name in glob.iglob(pattern, recursive=True):
        # Добавьте файл изображения в список файлов
        files.append(file_name)

    # Вернуть полный список файлов
    return files


def load_model(model_name):
    """
    Загрузка предварительно обученной модели обнаружения объектов и её сохранение на жестком диске.

    :param:str Name - имя предварительно обученной модели обнаружения объектов
    """
    url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/' + model_name + '.tar.gz'

    # Загрузка файла с URL-адреса, которого еще нет в кеше
    model_dir = tf.keras.utils.get_file(fname=model_name, untar=True, origin=url)

    print("Model path: ", str(model_dir))

    model_dir = str(model_dir) + "/saved_model"
    model = tf.saved_model.load(str(model_dir))

    return model


def load_rgb_images(pattern, shape=None):
    """
    Загружает изображения в формате RGB.

    :param:pattern str - паттерн имен файлов
    :param:shape - Размеры изображения (ширина, высота)
    """
    # Получим список всех файлов изображений в каталоге
    files = get_files(pattern)

    # Для каждого изображения в каталоге преобразуйте его из формата BGR в формат RGB.
    images = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB) for file in files]

    # Изменить размер изображения, если нужная форма предоставлена
    if shape:
        return [cv2.resize(img, shape) for img in images]
    else:
        return images


def load_ssd_coco():
    """
    Загрузка нейронной сети с архитектурой SSD, обученной на наборе данных COCO.
    """
    return load_model("ssd_resnet50_v1_fpn_640x640_coco17_tpu-8")


def save_image_annotated(img_rgb, file_name, output, model_traffic_lights=None):
    """
    Аннотируйте изображение с типами объектов и создавайте обрезанные изображения светофоров.
    """
    # Создать аннотированный файл изображения
    output_file = file_name.replace('.jpg', '_test.jpg')

    # Для каждой обнаруженной bounding box
    for idx in range(len(output['boxes'])):

        # Извлечь тип обнаруженного объекта
        obj_class = output["detection_classes"][idx]

        # Насколько модель обнаружения объектов уверена в типе объекта
        score = int(output["detection_scores"][idx] * 100)

        # Извлечь bounding box
        box = output["boxes"][idx]

        color = None
        label_text = ""

        if obj_class == LABEL_CAR:
            color = (255, 255, 0)
            label_text = "Car " + str(score)
        if obj_class == LABEL_BUS:
            color = (255, 255, 0)
            label_text = "Bus " + str(score)
        if obj_class == LABEL_TRUCK:
            color = (255, 255, 0)
            label_text = "Truck " + str(score)
        if obj_class == LABEL_TRAFFIC_LIGHT:
            color = (255, 255, 255)
            label_text = "Traffic Light " + str(score)

            if model_traffic_lights:

                # Аннотируйте изображение и сохраните его
                img_traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]
                img_inception = cv2.resize(img_traffic_light, (299, 299))

                # Надо раскомментить, чтобы сохранить обрезанное изображение светофора.
                # cv2.imwrite(output_file.replace('.jpg', '_crop.jpg'), cv2.cvtColor(img_inception, cv2.COLOR_RGB2BGR))
                img_inception = np.array([preprocess_input(img_inception)])

                prediction = model_traffic_lights.predict(img_inception)
                label = np.argmax(prediction)
                score_light = str(int(np.max(prediction) * 100))
                if label == 0:
                    label_text = "Green " + score_light
                elif label == 1:
                    label_text = "Yellow " + score_light
                elif label == 2:
                    label_text = "Red " + score_light
                else:
                    label_text = 'NO-LIGHT'  # это не светофор

        # Чтобы повысить производительность, лучше поиграться с порогом оценки score.
        # score находится на пороге от 0 до 100. Можно попробовать, например, 40.
        if color and label_text and accept_box(output["boxes"], idx, 5.0) and score > 50:
            cv2.rectangle(img_rgb, (box["x"], box["y"]), (box["x2"], box["y2"]), color, 2)
            cv2.putText(img_rgb, label_text, (box["x"], box["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(output_file, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print(output_file)


def center(box, coord_type):
    """
    Центр ограничивающей рамки.
    """
    return (box[coord_type] + box[coord_type + "2"]) / 2


def perform_object_detection(model, file_name, save_annotated=False, model_traffic_lights=None):
    """
    Выполняется обнаружение объектов на изображении с помощью предопределенной нейронной сети.
    """
    # Храним изображение
    img_bgr = cv2.imread(file_name)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img_rgb)  # input должен быть тензором
    input_tensor = input_tensor[tf.newaxis, ...]

    # Запускаем модель
    output = model(input_tensor)

    print("num_detections:", output['num_detections'], int(output['num_detections']))

    # Преобразуем тензоры в массив NumPy
    num_detections = int(output.pop('num_detections'))
    output = {key: value[0, :num_detections].numpy()
              for key, value in output.items()}
    output['num_detections'] = num_detections

    print('Detection classes:', output['detection_classes'])
    print('Detection Boxes:', output['detection_boxes'])

    # Обнаруженные классы должны быть integer
    output['detection_classes'] = output['detection_classes'].astype(np.int64)
    output['boxes'] = [
        {"y": int(box[0] * img_rgb.shape[0]), "x": int(box[1] * img_rgb.shape[1]), "y2": int(box[2] * img_rgb.shape[0]),
         "x2": int(box[3] * img_rgb.shape[1])} for box in output['detection_boxes']]

    if save_annotated:
        save_image_annotated(img_rgb, file_name, output, model_traffic_lights)

    return img_rgb, output, file_name


def perform_object_detection_video(model, video_frame, model_traffic_lights=None):
    """
    Выполняет обнаружение объектов на видео с помощью предопределенной нейронной сети.

    Возвращает аннотированный видеокадр.
    """
    # Храним изображение
    img_rgb = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(img_rgb)  # input должен быть тензором
    input_tensor = input_tensor[tf.newaxis, ...]

    # Запускаем модель
    output = model(input_tensor)

    # Преобразуем тензоры в массив NumPy
    num_detections = int(output.pop('num_detections'))
    output = {key: value[0, :num_detections].numpy()
              for key, value in output.items()}
    output['num_detections'] = num_detections

    # Обнаруженные классы должны быть integer
    output['detection_classes'] = output['detection_classes'].astype(np.int64)
    output['boxes'] = [
        {"y": int(box[0] * img_rgb.shape[0]), "x": int(box[1] * img_rgb.shape[1]), "y2": int(box[2] * img_rgb.shape[0]),
         "x2": int(box[3] * img_rgb.shape[1])} for box in output['detection_boxes']]

    # Для каждой обнаруженной bounding box
    for idx in range(len(output['boxes'])):

        # Извлечь тип обнаруженного объекта
        obj_class = output["detection_classes"][idx]

        # Насколько модель обнаружения объектов уверена в типе объекта
        score = int(output["detection_scores"][idx] * 100)

        # Извлечь bounding box
        box = output["boxes"][idx]

        color = None
        label_text = ""

        if obj_class == LABEL_CAR:
            color = (255, 255, 0)
            label_text = "Car " + str(score)
        if obj_class == LABEL_BUS:
            color = (255, 255, 0)
            label_text = "Bus " + str(score)
        if obj_class == LABEL_TRUCK:
            color = (255, 255, 0)
            label_text = "Truck " + str(score)
        if obj_class == LABEL_TRAFFIC_LIGHT:
            color = (255, 255, 255)
            label_text = "Traffic Light " + str(score)

            if model_traffic_lights:

                # Аннотируйте изображение и сохраните его
                img_traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]
                img_inception = cv2.resize(img_traffic_light, (299, 299))

                img_inception = np.array([preprocess_input(img_inception)])

                prediction = model_traffic_lights.predict(img_inception)
                label = np.argmax(prediction)
                score_light = str(int(np.max(prediction) * 100))
                if label == 0:
                    label_text = "Green " + score_light
                elif label == 1:
                    label_text = "Yellow " + score_light
                elif label == 2:
                    label_text = "Red " + score_light
                else:
                    label_text = 'NO-LIGHT'  # это не светофор

        # Используем переменную оценки, чтобы указать, насколько мы уверены, что это светофор (в %).
        # На реальном видеокадре мы отображаем достоверность того, что свет красный, зеленый,
        # желтый или недействительный светофор.
        if color and label_text and accept_box(output["boxes"], idx, 5.0) and score > 20:
            cv2.rectangle(img_rgb, (box["x"], box["y"]), (box["x2"], box["y2"]), color, 2)
            cv2.putText(img_rgb, label_text, (box["x"], box["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    output_frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return output_frame


def double_shuffle(images, labels):
    """
    Перемешивание изображений, чтобы добавить случайности.
    """
    indexes = np.random.permutation(len(images))

    return [images[idx] for idx in indexes], [labels[idx] for idx in indexes]


def reverse_preprocess_inception(img_preprocessed):
    """
    Обратный процесс предварительной обработки.
    """
    img = img_preprocessed + 1.0
    img = img * 127.5
    return img.astype(np.uint8)
