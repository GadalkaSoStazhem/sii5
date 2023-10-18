from my_knn import My_KNN
from preps import prep_data

def knn_result(X, y):
    X_train, X_test, y_train, y_test = prep_data(X, y)

    k3 = 3
    knn3 = My_KNN(k=k3)
    d1_y_train = y_train.values.reshape(1, y_train.shape[0])[0]
    knn3.fit(X_train.values, d1_y_train)
    y_pred3 = knn3.predict(X_test.values)

    print("Точность при k = 3: ", accuracy(y_test.values, y_pred3))
    print(conf_matrix(y_test.values, y_pred3))

    k5 = 5
    knn5 = My_KNN(k=k5)
    knn5.fit(X_train.values, d1_y_train)
    y_pred5 = knn5.predict(X_test.values)
    print("Точность при k = 5: ", accuracy(y_test.values, y_pred5))
    print(conf_matrix(y_test.values, y_pred5))

    k10 = 10
    knn10 = My_KNN(k=k10)
    knn10.fit(X_train.values, d1_y_train)
    y_pred10 = knn10.predict(X_test.values)
    print("Точность при k = 10: ", accuracy(y_test.values, y_pred10))
    print(conf_matrix(y_test.values, y_pred10))
def accuracy(test, test_prediction):
    correct = 0
    for i in range (len(test)):
        if test[i] == test_prediction[i]:
            correct += 1
    return (correct / len(test))

def conf_matrix(y_test, y_pred):
    cm = {
        'TP': 0,  # True Positives
        'TN': 0,  # True Negatives
        'FP': 0,  # False Positives
        'FN': 0   # False Negatives
    }
    for true, pred in zip(y_test, y_pred):
        if true == 1 and pred == 1:
            cm['TP'] += 1
        elif true == 0 and pred == 0:
            cm['TN'] += 1
        elif true == 0 and pred == 1:
            cm['FP'] += 1
        elif true == 1 and pred == 0:
            cm['FN'] += 1

    return cm