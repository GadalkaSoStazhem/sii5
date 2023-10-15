import os
import random
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from my_knn import My_KNN


# Путь к данным
data_path = 'C:\\Users\\Admin\\PycharmProjects\\sii5\\archive'
labels_file = 'C:\\Users\\Admin\\PycharmProjects\\sii5\\archive\\monkey_labels.txt'
training_data_path = os.path.join(data_path, 'training\\training')
validation_data_path = os.path.join(data_path, 'validation\\validation')

# Загрузка меток классов из файла
labels_df = pd.read_csv(os.path.join(data_path, labels_file))
classes= labels_df['Label'].tolist()
class_names = [c.strip() for c in classes]


# Функция для извлечения случайных признаков из изображения
def extract_random_features(image_path, num_features=10):
    try:
        image = cv2.imread(image_path)

        if image is not None:
            # Если изображение успешно загружено, выполните извлечение случайных признаков
            # Здесь вы можете добавить код для извлечения случайных признаков, например, выбор пикселей
            height, width, _ = image.shape
            random_features = []

            for _ in range(num_features):
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                pixel = image[y, x]
                random_features.extend(pixel)

            return random_features
        else:
            # Если изображение не было загружено (None), верните None как признаки
            return None
    except Exception as e:
        # Обработайте возможные ошибки, связанные с изображением
        print(f"Ошибка при обработке изображения: {str(e)}")
        return None


# Случайно выбираем признаки изображений
random_features = []
random_labels = []

for class_name in class_names:
    class_path = training_data_path + "\\" + class_name + "\\"
    #class_path = os.path.join(training_data_path, class_name, "\\")
    image_filenames = os.listdir(class_path)
    for image_filename in image_filenames:
        image_path = os.path.join(class_path, image_filename)
        features = extract_random_features(image_path)

        if features is not None:
            random_features.append(features)
            random_labels.append(class_name)


# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(random_features, random_labels, test_size=0.2, random_state=42)

# Обучение KNN классификатора


my_knn = My_KNN(k = 5)
my_knn.fit(X_train, y_train)

# Тестирование классификатора

y_pred_my = my_knn.predict(X_test)
# Оценка производительности

acc = accuracy_score(y_test, y_pred_my)

print(f'Точность классификации: {acc * 100:.2f}%')




