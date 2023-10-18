from preps import *
from features import features_choice, show_corr_matrix
from model_work import *
df = pd.read_csv("diabetes.csv")

nan_check(df)
checked_df = cat_features(df)
#define_distrib(checked_df)
#show_corr_matrix(df)
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_rand, X_nrand = features_choice(X)
print("Рандомный результат:")
knn_result(X_rand, y)
print("\nНе рандомный результат:")
knn_result(X_nrand, y)




