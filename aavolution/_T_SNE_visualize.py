import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def tsne_visualize(job_name,
                   df: pd.DataFrame,
                   pred_dict_low_border: dict,
                   pred_color_dict: dict,
                   df_plus: pd.DataFrame = None,
                   max_instance_per_category: int = 300):

    # initialize T-SNE
    tsne = TSNE(n_components=2, learning_rate=30, n_iter=1500, n_jobs=-1, perplexity=40)
    # configuring the parameters
    # the number of components = 2
    # default perplexity = 30
    # default learning rate = 200S
    # default Maximum number of iterations
    # for the optimization = 1000

    # data transformation (df, label_dict_low_border)
    # __________________________________________________________________________________________________________________
    # sort dictionary decending and transform to list
    keys = list(pred_dict_low_border.keys())
    sorted_keys = sorted(keys, key=lambda x: float(x), reverse=True)
    list_key_value = [[key, pred_dict_low_border[key]] for key in sorted_keys]

    # generate the new label list
    label_list = df["pred"].tolist()
    new_label = []
    for label in label_list:
        counter = 0
        while True:
            if label > list_key_value[counter][0]:
                new_label.append(list_key_value[counter][1])
                break
            counter += 1

    df_transform_pre = df.drop("pred", axis=1)
    df_transform_pre["pred"] = new_label

    # limit each category to a set max of instances (max_instance_per_category)
    unique_label_new = [tag[1] for tag in list_key_value]
    list_df_slices = []
    for unique in unique_label_new:
        if max_instance_per_category > df_transform_pre[df_transform_pre["pred"] == unique].shape[0]:
            list_df_slices.append(df_transform_pre[df_transform_pre["pred"] == unique])
        else:
            list_df_slices.append(df_transform_pre[df_transform_pre["pred"] == unique]
                                  .sample(n=max_instance_per_category, replace=False))
    df_transform_pre = pd.concat(list_df_slices, axis=0)

    # for adding a special comparison DataFrame (SUBSTRATES and NONSUBSTRATES)
    if df_plus is not None:
        df_transform_pre = pd.concat([df_transform_pre, df_plus], axis=0)

    # final dataframes
    # __________________________________________________________________________________________________________________
    df_transform_labels = df_transform_pre["pred"]
    df_transform = df_transform_pre.drop("pred", axis=1)

    # tsne
    # __________________________________________________________________________________________________________________
    tsne_data = tsne.fit_transform(df_transform)

    # creating a new data frame which
    # help us in plotting the result data
    tsne_data = np.vstack((tsne_data.T, df_transform_labels)).T
    tsne_df = pd.DataFrame(data=tsne_data,
                           columns=("dimension 1", "dimension 2", "pred"))

    # Plotting the result of tsne
    sns.scatterplot(data=tsne_df, x='dimension 1', y='dimension 2',
                    hue='pred', palette=pred_color_dict)
    sns.despine(offset=10, trim=True)
    plt.legend(loc="upper right")
    plt.savefig(f"T-SNE_{job_name}", bbox_inches="tight", dpi=300)
    plt.show()


# run T-SNE
# ______________________________________________________________________________________________________________________
if __name__ == "__main__":
    job_name = "W8_nonsub_plus_better_propensity"
    data = pd.read_excel("/home/freiherr/PycharmProjects/AAvolution/aavolution/W8_TSNE Advanced/W8 RUN NONSUB plus_2024-04-22/W8 RUN NONSUB plus_generated_sequences_features_preds.xlsx").drop("Unnamed: 0", axis=1)
    data_lab = pd.read_excel("/home/freiherr/PycharmProjects/AAvolution/aavolution/W8_TSNE Advanced/W8 RUN NONSUB plus_2024-04-22/W8 RUN NONSUB plus_bench_features_preds.xlsx").set_index("entry")

    # borders
    dict_name = {0.965152659344041: "top 100",
                 0.95: "> 95%",
                 0.80: "> 80%",
                 0.50: "> 50%",
                 0.20: "> 20%",
                 0.00: ">= 0%"}

    # gradient color scheme + color highlights for lab data
    dict_color = {"top 100": "#f1f426",
                  "> 95%": "#f5b25f",
                  "> 80%": "#f0804e",
                  "> 50%": "#c6417d",
                  "> 20%": "#8707a6",
                  ">= 0%": "#110788",
                  "SUBEXP": "#23cdd9",
                  "SUBLIT": "#166fab",
                  "NONSUB": "#ed2125"}

    tsne_visualize(job_name, data, dict_name, dict_color, data_lab, 200)
