# general imports
import StandardConfig as stdc
from StandardConfig import timingmethod
from datetime import date
import pandas as pd
import numpy as np
import copy
# ml imports
from AAvolver import seq_splitter, seq_agglomerater, evolution_display
from AAvolver import AAvolutionizer as aav
import aaanalysis as aa
from ML_AAvolution import StackedClassifiers as sc


@timingmethod
def run_aavolution(job_name: str,
                   parts: list,
                   mode: str,
                   df_seq_train: pd.DataFrame,
                   df_feat_train: pd.DataFrame,
                   df_bench_pred: pd.DataFrame = None,
                   propensity_increment_display: int = 10,  # for AAlogo
                   dict_parts: dict = {},
                   dict_evo_params: dict = {},
                   raw_data_only=False):
    path_file, path_module, sep = stdc.find_folderpath()
    folder_path = f"{path_file}{sep}{job_name}_{date.today()}"
    if raw_data_only is False:                                  # EXPORT
        stdc.make_directory(f"{job_name}_{date.today()}")
    # check inputs
    # __________________________________________________________________________________________________________________

    # set up model
    # __________________________________________________________________________________________________________________
    sf = aa.SequenceFeature()
    df_parts = sf.get_df_parts(df_seq=df_seq_train)
    train_x = sf.feature_matrix(features=df_feat_train["feature"], df_parts=df_parts, accept_gaps=True)
    labels = df_seq_train["label"].to_list()

    # stacked predicters (ML_AAvolution)
    clfs_model = sc.clfs_fit(train_x, labels)

    if isinstance(df_bench_pred, pd.DataFrame):
        sf_bench = aa.SequenceFeature()
        df_parts = sf_bench.get_df_parts(df_seq=df_bench_pred)
        bench_x = sf.feature_matrix(features=df_feat_train["feature"], df_parts=df_parts, accept_gaps=True)
        pred_bench = [sublist[1] for sublist in clfs_model.clfs_proba(bench_x)]

        # save benchmark features
        bench_feature_df = pd.DataFrame(bench_x)
        bench_feature_df["entry"] = df_bench_pred["entry"].tolist()
        bench_feature_df["pred"] = pred_bench
        if raw_data_only is False:                                  # EXPORT
            bench_feature_df.set_index("entry").to_excel(f"{folder_path}{sep}{job_name}_bench_features_preds.xlsx")

    else:
        pred_bench = [sublist[1] for sublist in clfs_model.clfs_proba(train_x)]

    if "P05067" in df_bench_pred.set_index("entry").index.tolist():
        APP_pos = df_bench_pred.set_index("entry").index.tolist().index("P05067")
        APP_value = pred_bench[APP_pos]
    else:
        APP_pos = None
        APP_value = None

    mean_bench, max_bench = sum(pred_bench)/len(pred_bench), max(pred_bench)

    # run
    # __________________________________________________________________________________________________________________
    population_size = 200
    if "set_population_size" in dict_evo_params.keys():
        if isinstance(dict_evo_params["set_population_size"], int):
            if dict_evo_params["set_population_size"] >= 1:
                population_size = dict_evo_params["set_population_size"]

    max_generation = 500
    if "max_gen" in dict_evo_params.keys():
        if isinstance(dict_evo_params["max_gen"], int):
            if dict_evo_params["max_gen"] >= 1:
                max_generation = dict_evo_params["max_gen"]

    initialize = aav.gen_zero_maker(mode=mode, init_population=population_size, **dict_parts)
    initialize.set_mut_params(**dict_evo_params)
    pool_df = initialize.return_parts()
    list_seq_gens = []
    list_mean = []
    list_metrics_gens = []
    list_sf_pred = []
    gen_counter = 1
    mut_cycle = 1
    while mut_cycle <= initialize.MAX_GENERATIONS:
        print(f"Current Generation: {mut_cycle}/{max_generation}")
        child_sf = aa.SequenceFeature()
        # aaanalysis requires for seq and "entry" column
        child_df_parts = child_sf.get_df_parts(df_seq=pool_df.reset_index().rename(columns={"index": "entry"}))
        child_x = sf.feature_matrix(features=df_feat_train["feature"], df_parts=child_df_parts)
        # predicting the offspring
        child_pred = [sublist[1] for sublist in clfs_model.clfs_proba(child_x)]
        pool_df["pred"] = child_pred
        # create dataframe list with sequence features
        child_feature_df = pd.DataFrame(child_x)
        child_feature_df["pred"] = child_pred
        list_sf_pred.append(child_feature_df)

        # save data
        mean_gen, max_gen = sum(child_pred) / len(child_pred), max(child_pred)
        list_mean.append(mean_gen)
        list_metrics_gens.append([max_gen, mean_gen])
        pool_df_decending = pool_df.sort_values("pred", ascending=False)
        list_seq_gens.append(pool_df_decending)

        if raw_data_only is False:  # EXPORT
            if gen_counter == propensity_increment_display:
                stdc.make_directory(f"generations sequences (increment: {propensity_increment_display})", folder_path)
                pool_df_decending.to_excel(f"{folder_path}{sep}generations sequences (increment: {propensity_increment_display}){sep}seq_gen_{mut_cycle}.xlsx")
                gen_counter = 1
            else:
                gen_counter += 1

        # select the top
        # pool_df_selected = pool_df_decending.head(30)
        pool_df_selected = pool_df_decending[pool_df_decending["pred"] > mean_gen]
        # mating
        new_gen = initialize.mate_survivors(pool_df_selected[parts])
        # prediction
        # split for mutation
        split_pool = seq_splitter(new_gen, parts)
        split_pool_for_mut = copy.deepcopy(split_pool)
        # mutation cycle
        mut_pool = aav.single_point_mutation(split_pool_for_mut)
        new_mut_list = copy.deepcopy(mut_pool)
        cross_pool = aav.crossover_allel(new_mut_list)
        new_cross_pool = copy.deepcopy(cross_pool)
        indel_pool = initialize.indels_mutation(new_cross_pool)
        agg_pool = seq_agglomerater(indel_pool, parts)
        tmd_filter = initialize.tmd_length_filter(agg_pool)

        # mating
        pool_df = initialize.mate_survivors(tmd_filter[parts])
        mut_cycle += 1

    # top 100
    top100 = pd.concat(list_seq_gens, axis=0).sort_values("pred", ascending=False).drop_duplicates().head(100)
    max_mean_gens = pd.DataFrame(list_metrics_gens, columns=["max", "mean"])

    if raw_data_only:                                  # EXPORT
        return job_name, max_mean_gens, mean_bench, max_bench, APP_value
    else:
        evolution_display(list_pred_dfs=max_mean_gens,
                          job_name=job_name,
                          mean_benchmark=mean_bench,
                          max_benchmark=max_bench,
                          APP_benchmark=APP_value,
                          set_path=f"{folder_path}{sep}")
        top100.to_excel(f"{folder_path}{sep}{job_name}_top100.xlsx")
        all_child_features = pd.concat(list_sf_pred, axis=0)
        all_child_features.to_excel(f"{folder_path}{sep}{job_name}_generated_sequences_features_preds.xlsx")
        return top100


# script part
# ______________________________________________________________________________________________________________________
if __name__ == "__main__":
    job_name = "W8 RUN NONSUB plus"
    dict_evo_settings = {"set_population_size": 500,
                         "max_gen": 200,
                         "n_point_mut": 4,
                         "n_crossover_per_seg": 2,
                         "n_indels": 2}

    path_test = "/home/freiherr/PycharmProjects/AAvolution/_test"
    path_data = "/home/freiherr/PycharmProjects/AAvolution/aavolution/w8_uniprot_refining"
    test_feat = pd.read_excel(f"{path_data}/CPP_feat_N8_SUB.xlsx")
    subexp_df = pd.read_excel(f"{path_data}/w8_TMD_refined.xlsx", "SUBEXP")
    sublit_df = pd.read_excel(f"{path_data}/w8_TMD_refined.xlsx", "SUBLIT")
    nonsub_df = pd.read_excel(f"{path_data}/w8_TMD_refined.xlsx", "NONSUB")
    test_seq = pd.concat([subexp_df, sublit_df, nonsub_df], axis=0).reset_index().drop("index", axis=1)
    nonsub_test = pd.read_excel("/home/freiherr/PycharmProjects/AAvolution/aavolution/W8_TSNE Advanced/top1000 NONSUB_2024-04-22/top1000 NONSUB_top100.xlsx")
    bench_seq = pd.concat([subexp_df, sublit_df], axis=0).reset_index().drop("index", axis=1)
    bench_seq_nonsub_plus = pd.concat([test_seq[["entry", "jmd_n", "tmd", "jmd_c"]], nonsub_test], axis=0)
    print(test_seq)
    top100_test = run_aavolution(job_name=job_name,
                                 mode="prop_NON_n_SUB",
                                 parts=["jmd_n", "tmd", "jmd_c"],
                                 df_seq_train=test_seq,
                                 df_feat_train=test_feat,
                                 df_bench_pred=bench_seq_nonsub_plus,
                                 propensity_increment_display=1,
                                 dict_evo_params=dict_evo_settings,
                                 raw_data_only=False)
    print(top100_test)
