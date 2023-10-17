import pandas as pd
import matplotlib.pyplot as plt
from characteristics import mean_vals, st_deviation
class LabelEncoder:
    def __init__(self):
        self.label_map = {}
        self.inverse_label_map = {}

    def fit(self, labels):
        unique_labels = set(labels)
        for i, label in enumerate(unique_labels):
            self.label_map[label] = i
            self.inverse_label_map[i] = label

    def transform(self, labels):
        return [self.label_map[label] for label in labels]

    def inverse_transform(self, encoded_labels):
        return [self.inverse_label_map[label] for label in encoded_labels]


def nan_check(df):
    missed = df.isna().sum().sum()
    if missed > 0:
        print("Количество пропущенных значений: ")
        for col in df.columns:
            if type(df[col].to_dict()[0]) == int or type(df[col].to_dict()[0]) == float:
                df[col].fillna(df[col].mean(), inplace = True)
            else:
                df[col].fillna(df[col].mode().iloc[0], inplace = True)
    else:
        print("Пропущенных значений нет")

def cat_features (df):
    flag = 0
    for col in df.columns:
        if type(df[col].to_dict()[0]) == str:
            flag = 1
            cat_col = pd.get_dummies(df[col],  drop_first=True, dtype=int)
            to_add = pd.DataFrame(data = cat_col.values, columns=[df[col].name])
            df = df.drop(df[col].name, axis = 1)
            df_good = pd.concat([df, to_add], axis = 1)


    if flag == 0:
        print("Категориальных признаков нет")
        return df
    else:
        print("Категориальные признаки изменены")
        return df_good

def define_distrib(df):
    plt.figure(figsize=(20, 10))
    for i, col in enumerate(df.columns):
        plt.subplot(3, 3, i + 1)
        df[col].plot(kind='hist')
        plt.title(df[col].name)
    plt.tight_layout()
    plt.show()

def std_scaler(df):
    cnt = 0
    shape = df.shape
    means = mean_vals(df, shape[0])
    devs = st_deviation(df, means, shape[0])
    #используется для нормального и геометрического распределения
    for col in df.columns:
        if df[col].name != 'Outcome':
            df[col] = (df[col] - means[cnt]) / devs[cnt]
        cnt += 1

def min_max_scaler(df):
    #используется для равномерного распределения
    for col in df.columns:
        if df[col].name == 'Outcome':
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)


