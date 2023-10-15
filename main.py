import os
import random
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Путь к данным
data_path = 'archive'
labels_file = 'monkey_labels.txt'

# Загрузка меток классов из файла
labels_df = pd.read_csv(os.path.join(data_path, labels_file))
class_names = labels_df['Label'].tolist()
num_classes = len(class_names)


# Функция для извлечения признаков из изображения
def extract_features(image_path):
    # Здесь вы можете добавить код для извлечения признаков, например, гистограмм цветов
    pass


# Случайно выбираем признаки изображений
random_features = []
random_labels = []

for class_name in class_names:
    class_path = os.path.join(data_path, 'training', class_name)
    image_filenames = os.listdir(class_path)

    # Случайно выбираем некоторое количество изображений
    num_images_to_use = 10  # Задайте количество изображений
    random_images = random.sample(image_filenames, num_images_to_use)

    for image_filename in random_images:
        image_path = os.path.join(class_path, image_filename)
        features = extract_features(image_path)
        random_features.append(features)
        random_labels.append(class_name)

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(random_features, random_labels, test_size=0.2, random_state=42)

# Обучение KNN классификатора
k = 5  # Количество соседей
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Тестирование классификатора
y_pred = knn.predict(X_test)

# Оценка производительности
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность классификации: {accuracy * 100:.2f}%')



