import collections  # Handles specialized container datatypes
import cv2  # Computer vision library
import matplotlib.pyplot as plt  # Plotting library
import numpy as np  # Scientific computing library
import object_detection  # Custom object detection program
import sys
import tensorflow as tf  # Machine learning library
from tensorflow import keras  # Library for neural networks
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

sys.path.append('../')

print("TensorFlow", tf.__version__)
print("Keras", keras.__version__)


def show_history(history):
    """
    Визуализация истории обучения модели нейронной сети
    :param:history — запись значений потерь при обучении и значений
           показателей в последовательные эпохи, а также значения потерь
           при проверке и значения показателей проверки.
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    plt.show()


def Transfer(n_classes, freeze_layers=True):
    """
    Используется архитектура нейронной сети InceptionV3 для выполнения трансферного обучения.

    :param:n_classes - кол-во классов
    :param:freeze_layers - если True, параметры сети не изменяются
    :return Лучшая нейронная сеть
    """
    print("Loading Inception V3...")

    # Первый результат поиска должен отправить вас на веб-сайт Keras, на котором есть объяснение того,
    # что означает каждый из этих параметров.
    # include_top означает, что мы удаляем верхнюю часть начальной модели, которая является классификатором.
    # input_shape должен иметь 3 канала и разрешение не менее 75x75.
    # Наша нейронная сеть будет построена на основе модели Inception V3 (обученной на наборе данных ImageNet).
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    print("Inception V3 has finished loading.")

    # Показать базовую сетевую архитектуру
    print('Layers: ', len(base_model.layers))
    print("Shape:", base_model.output_shape[1:])
    print("Shape:", base_model.output_shape)
    print("Shape:", base_model.outputs)
    base_model.summary()

    # Создаем нейронную сеть. В этой сети используется последовательная архитектура,
    # в которой каждый слой имеет один входной тензор (например, вектор, матрицу и т. д.)
    # и один выходной тензор.
    top_model = Sequential()

    # Наша модель классификатора будет построена поверх базовой модели.
    top_model.add(base_model)
    top_model.add(GlobalAveragePooling2D())
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1024, activation='relu'))
    top_model.add(BatchNormalization())
    top_model.add(Dropout(0.5))
    top_model.add(Dense(512, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dense(n_classes, activation='softmax'))

    # Заморозим слои в модели, чтобы их нельзя было обучить (т.е. параметры в нейросети не изменятся)
    if freeze_layers:
        for layer in base_model.layers:
            layer.trainable = False

    return top_model


# Выполняем аугментацию изображения.
# Аугментация изображения позволяет нам изменять доступные изображения
# (например: поворот, переворот, изменение оттенка и т.д.), чтобы генерировать больше изображений,
# которые наша нейронная сеть может использовать для обучения,
# тем самым избавляя нас от необходимости собирать больше внешних изображений.
datagen = ImageDataGenerator(rotation_range=5, width_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                             zoom_range=[0.7, 1.5], height_shift_range=[-10, -5, -2, 0, 2, 5, 10],
                             horizontal_flip=True)

shape = (299, 299)

# Загрузим обрезанные изображения светофора из соответствующего каталога.
img_0_green = object_detection.load_rgb_images("traffic_light_dataset/0_green/*", shape)
img_1_yellow = object_detection.load_rgb_images("traffic_light_dataset/1_yellow/*", shape)
img_2_red = object_detection.load_rgb_images("traffic_light_dataset/2_red/*", shape)
img_3_not_traffic_light = object_detection.load_rgb_images("traffic_light_dataset/3_not/*", shape)

# Создаем список меток той же длины, что и количество изображений в каждой категории
# 0 = green
# 1 = yellow
# 2 = red
# 3 = не светофор
labels = [0] * len(img_0_green)
labels.extend([1] * len(img_1_yellow))
labels.extend([2] * len(img_2_red))
labels.extend([3] * len(img_3_not_traffic_light))

# Создаем массив NumPy
labels_np = np.ndarray(shape=(len(labels), 4))
images_np = np.ndarray(shape=(len(labels), shape[0], shape[1], 3))

# Создаем список всех изображений из dataset светофоров
img_all = []
img_all.extend(img_0_green)
img_all.extend(img_1_yellow)
img_all.extend(img_2_red)
img_all.extend(img_3_not_traffic_light)

# Нужно убедиться, что у нас столько же изображений, сколько у нас меток
assert len(img_all) == len(labels)

# Перемешаем изображения
img_all = [preprocess_input(img) for img in img_all]
(img_all, labels) = object_detection.double_shuffle(img_all, labels)

# Храним изображения и метки в массиве NumPy
for idx in range(len(labels)):
    images_np[idx] = img_all[idx]
    labels_np[idx] = labels[idx]

print("Images: ", len(img_all))
print("Labels: ", len(labels))

# Выполняем one-hot кодирование
for idx in range(len(labels_np)):
    # У нас есть четыре целочисленных метки, представляющих разные цвета светофоров.
    labels_np[idx] = np.array(to_categorical(labels[idx], 4))

# Разделяем набор данных на training набор и validation набор.
# Training набор — это часть набора данных, которая используется для определения параметров
# (например, весов) нейронной сети.
# Validation набор — это часть набора данных, используемая для точной настройки параметров модели
# (т. е. гиперпараметров), которые фиксируются перед обучением и тестированием нейронной сети на данных.
# Набор проверки помогает нам выбрать окончательную модель (например, скорость обучения,
# количество скрытых слоев, количество скрытых юнитов, функции активации, количество эпох и т.д.)
# В этом случае 80% набора данных становятся обучающими данными, а 20% набора данных становятся проверочными данными.
idx_split = int(len(labels_np) * 0.8)
x_train = images_np[0:idx_split]
x_valid = images_np[idx_split:]
y_train = labels_np[0:idx_split]
y_valid = labels_np[idx_split:]

# Храним подсчет количества светофоров каждого цвета
cnt = collections.Counter(labels)
print('Labels:', cnt)
n = len(labels)
print('0:', cnt[0])
print('1:', cnt[1])
print('2:', cnt[2])
print('3:', cnt[3])

# Рассчитаем вес каждого класса светофора
class_weight = {0: n / cnt[0], 1: n / cnt[1], 2: n / cnt[2], 3: n / cnt[3]}
print('Class weight:', class_weight)

# Сохраним лучшую модель как traffic.h5
checkpoint = ModelCheckpoint("traffic.h5", monitor='val_loss', mode='min', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(min_delta=0.0005, patience=15, verbose=1)

# Генерируем модель с использованием трансферного обучения
model = Transfer(n_classes=4, freeze_layers=True)

# Отобразим сводки модели нейронной сети
model.summary()

# Генерируем пакет (batch) случайно преобразованных изображений
it_train = datagen.flow(x_train, y_train, batch_size=32)

# Настроим параметры модели для обучения
model.compile(loss=categorical_crossentropy, optimizer=Adadelta(
    lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0), metrics=['accuracy'])

# Обучаем модель на пакетах изображений для фиксированного количества эпох.
# Сохраняем запись об ошибке в наборе обучающих данных и значениях метрик в объекте истории.
# 250 эпох -> 100 эпох
history_object = model.fit(it_train, epochs=100, validation_data=(
    x_valid, y_valid), shuffle=True, callbacks=[
    checkpoint, early_stopping], class_weight=class_weight)

# Выведем историю тренинга
show_history(history_object)

# Получим значение потерь (loss) и значения метрик в наборе проверочных данных.
score = model.evaluate(x_valid, y_valid, verbose=0)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

print('Saving the validation data set...')

print('Length of the validation data set:', len(x_valid))

# Просмотрим набор проверочных данных и увидим, как модель работала на каждом изображении.
for idx in range(len(x_valid)):
    # Сделаем изображение массивом NumPy
    img_as_ar = np.array([x_valid[idx]])

    # Генерация прогнозов
    prediction = model.predict(img_as_ar)

    # Определим, какая метка основана на наибольшей вероятности
    label = np.argmax(prediction)

    # Создадим папку и файл для набора данных проверки.
    # После каждого запуска удаляем этот каталог out_valid/, чтобы там не болтались старые файлы.
    file_name = str(idx) + "_" + str(label) + "_" + str(np.argmax(str(y_valid[idx]))) + ".jpg"
    img = img_as_ar[0]

    # Обратный процесс предварительной обработки изображения
    img = object_detection.reverse_preprocess_inception(img)

    # Сохраним изображение
    cv2.imwrite(file_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

print('The validation data set has been saved!')
