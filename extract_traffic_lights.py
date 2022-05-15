import cv2  # Computer vision library
import object_detection  # Contains methods for object detection in images

# Получим список файлов изображений JPEG, содержащих светофоры
files = object_detection.get_files('traffic_light_input/*.jpg')

# Загрузим модель обнаружения объектов
this_model = object_detection.load_ssd_coco()

# Следим за количеством найденных светофоров
traffic_light_count = 0

# Следим за количеством файлов изображений, которые были обработаны
file_count = 0

# Отобразим количество изображений, которые нам нужно обработать
print("Number of Images:", len(files))

# Просмотрим каждый файл изображения по одному
for file in files:

    # img_rgb — исходное изображение в формате RGB
    # out — словарь, содержащий результаты обнаружения объектов
    # file_name — имя файла
    (img_rgb, out, file_name) = object_detection.perform_object_detection(model=this_model, file_name=file,
                                                                          save_annotated=None,
                                                                          model_traffic_lights=None)

    # Каждые 10 обрабатываемых файлов
    if (file_count % 10) == 0:
        # Отобразить количество обработанных файлов
        print("Images processed:", file_count)

        # Отображение общего количества светофоров, которые были идентифицированы до сих пор
        print("Number of Traffic lights identified: ", traffic_light_count)

    # Увеличиваем кол-во файлов на 1
    file_count = file_count + 1

    # Для каждого обнаруженного светофора (т. е. ограничивающего прямоугольника)
    for idx in range(len(out['boxes'])):

        # Извлечь тип обнаруженного объекта
        obj_class = out["detection_classes"][idx]

        # Если обнаруженный объект является светофором
        if obj_class == object_detection.LABEL_TRAFFIC_LIGHT:
            # Извлечь координаты bounding box
            box = out["boxes"][idx]

            # Извлечь (т.е. обрезать) светофор из изображения
            traffic_light = img_rgb[box["y"]:box["y2"], box["x"]:box["x2"]]

            # Преобразование светофора из формата RGB в формат BGR
            traffic_light = cv2.cvtColor(traffic_light, cv2.COLOR_RGB2BGR)

            # Сохранить обрезанное изображение в папке с именем 'traffic_light_cropped'.
            cv2.imwrite("traffic_light_cropped/" + str(traffic_light_count) + ".jpg", traffic_light)

            # Увеличить количество светофоров на 1
            traffic_light_count = traffic_light_count + 1

# Отображение общего количества идентифицированных светофоров
print("Number of Traffic lights identified:", traffic_light_count)
