import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def tsne_visualize(df: pd.DataFrame, label_dict_low_border: dict, max_instance_per_category: int = 300):
    tsne = TSNE(n_components=2)
    # configuring the parameters
    # the number of components = 2
    # default perplexity = 30
    # default learning rate = 200
    # default Maximum number of iterations
    # for the optimization = 1000

    # data transformation (df, label_dict_low_border)
    # __________________________________________________________________________________________________________________
    # sort dictionary decending and transform to list
    keys = list(label_dict_low_border.keys())
    sorted_keys = sorted(keys, key=lambda x: float(x), reverse=True)
    list_key_value = [[key, label_dict_low_border[key]] for key in sorted_keys]

    # generate the new label list
    label_list = df["label"].tolist()
    new_label = []
    for label in label_list:
        counter = 0
        while True:
            if label >= list_key_value[counter][0]:
                new_label.append(list_key_value[counter][1])
                break
            counter += 1

    df_transform_pre = df.drop("label", axis=1)
    df_transform_pre["label"] = new_label

    # limit each category to a set max of instances (max_instance_per_category)
    unique_label_new = list(set(new_label))
    list_df_slices = []
    for unique in unique_label_new:
        list_df_slices.append(df_transform_pre[df_transform_pre["label"] == unique].sample(n=max_instance_per_category,
                                                                                           replace=False))
    df_transform_pre = pd.concat(list_df_slices, axis=1)

    # final dataframes
    # __________________________________________________________________________________________________________________
    df_transform_labels = df_transform_pre["label"]
    df_transform = df_transform_pre.drop("labels", axis=1)

    # tsne
    # __________________________________________________________________________________________________________________
    tsne_data = tsne.fit_transform(df_transform)

    # creating a new data frame which
    # help us in plotting the result data
    tsne_data = np.vstack((tsne_data.T, df_transform_labels)).T
    tsne_df = pd.DataFrame(data=tsne_data,
                           columns=("Dim_1", "Dim_2", "label"))

    # Plotting the result of tsne
    sns.scatterplot(data=tsne_df, x='Dim_1', y='Dim_2',
                    hue='label', palette="rocket")
    sns.despine(offset=10, trim=True)
    plt.show()
