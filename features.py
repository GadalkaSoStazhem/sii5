import seaborn as sns
import matplotlib.pyplot as plt
import random

def get_rand_ftr(col_names_list):
    amount = random.randint(1, len(col_names_list))
    ids = list(range(0, amount))
    random.shuffle(ids)
    col_chosen = []
    for i in ids:
        col_chosen.append(col_names_list[i])
    return col_chosen

def get_rand_frame(df):
    col_chosen = get_rand_ftr(df.columns)
    new_df = df[col_chosen]
    print("Количество признаков: ", len(col_chosen))
    print("Выбранные рандомно признаки", col_chosen)
    return new_df

def show_corr_matrix(X_train):
    corred = X_train.corr().round(2)
    sns.heatmap(corred, annot = True)
    plt.show()

def get_not_rand_frame(df):
    n_x = df.drop(['Age', 'Pregnancies'], axis=1)
    print("Выбранные не рандомно признаки: ", n_x.columns)
    return n_x

def features_choice(df):
    r_df = get_rand_frame(df)
    nr_df = get_not_rand_frame(df)
    return r_df, nr_df