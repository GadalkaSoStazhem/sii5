import seaborn as sns
import matplotlib.pyplot as plt
def show_corr_marix(X_train):
    corred = X_train.corr().round(2)
    sns.heatmap(corred, annot = True)
    plt.show()