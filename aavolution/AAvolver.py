# general util import
import pandas as pd
from StandardConfig import timingmethod
import StandardConfig as stdc
import numpy as np
# ML tools import
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import aaanalysis as aa
# visualize
import matplotlib.pyplot as plt
import seaborn as sns


# script and module path! + seperator depending on operating system
# ______________________________________________________________________________________________________________________
path_script, path_module, sep = stdc.find_folderpath()

# import necessary tables as DataFrame
# ______________________________________________________________________________________________________________________
path_data_sheets = f"{path_module.split('aavolution')[0]}_data{sep}"
aa_propensities_df = pd.read_excel(f"{path_data_sheets}aa_propensities.xlsx").set_index("AA")
length_dist_df = pd.read_excel(f"{path_data_sheets}length_dist.xlsx").set_index("len_tmd")


def gen_zero_maker(mode: str = "rand_prop", **parts_kwargs):
    # check inputs
    # __________________________________________________________________________________________________________________
    # check kwargs
    allowed_parts = ["jmd_n", "tmd", "jmd_c"]
    set_part_slices = ["jmd_n", "tmd", "jmd_c"]
    if "set_part_slices" in parts_kwargs.keys():
        if not isinstance(parts_kwargs["set_part_slices"], list):
            raise TypeError("set_part_slices must be a list!")
        for tags in parts_kwargs["set_part_slices"]:
            if tags not in allowed_parts:
                raise ValueError(f"not allowed part in list, allowed parts: {allowed_parts}")
        set_part_slices = parts_kwargs["set_part_slices"]

    set_len_part_slices = [10, 0, 10]  # if < 1 ==> no set length, dependent on distribution given by mode
    if "set_len_part_slices" in parts_kwargs.keys():
        if not isinstance(parts_kwargs["set_len_part_slices"], list):
            raise TypeError("set_len_part_slices must be a list!")
        for tags in parts_kwargs["set_len_part_slices"]:
            if not isinstance(tags, int):
                raise TypeError(f"not allowed type, only enter ints")
        set_len_part_slices = parts_kwargs["set_len_part_slices"]

    # check if both lists have same length
    if len(set_part_slices) != len(set_len_part_slices):
        raise ValueError(f"parts list ({len(set_part_slices)}) and parts length list ({len(set_len_part_slices)}) are not of same length!")

    # connect mode with correct table entries: aa_propensity, length_dist
    dict_mode = {"rand_prop": {"jmd_n": "codon_table",
                               "tmd": "codon_table",
                               "jmd_c": "codon_table"},
                 "prop_N_out": {"jmd_n": "N_out_JMD_N",
                                "tmd": "N_out_TMD",
                                "jmd_c": "N_out_JMD_C"},
                 "prop_SUB": {"jmd_n": "SUB_JMD_N",
                              "tmd": "SUB_TMD",
                              "jmd_c": "SUB_JMD_C"}}

    if mode not in dict_mode.keys():
        mode = "rand_prop"
    seq_parts_list = []
    for part_seq, len_part in zip(set_part_slices, set_len_part_slices):

        raw_aa = aa_propensities_df[dict_mode[mode][part_seq]]
        # properties of length
        if mode == "prop_N_out":
            if len_part < 1:
                length_aa_series = length_dist_df[dict_mode[mode][part_seq]].dropna()
                pre_p = length_aa_series.to_numpy().tolist()
                norm_p = [p/sum(pre_p) for p in pre_p]
                length_aa = np.random.choice(length_aa_series.index.tolist(), 1, p=norm_p)
            else:
                length_aa = len_part
        elif mode == "prop_SUB":
            if len_part < 1:
                length_aa_series = length_dist_df[dict_mode[mode][part_seq]].dropna()
                pre_p = length_aa_series.to_numpy().tolist()
                norm_p = [p/sum(pre_p) for p in pre_p]
                length_aa = np.random.choice(length_aa_series.index.tolist(), 1, p=norm_p)
            else:
                length_aa = len_part
        else:
            if len_part < 1:
                length_aa = np.random.randint(15, 35)  # based on TMHMM arbitrary rules
            else:
                length_aa = len_part

        list_aa_letters = aa_propensities_df.index.tolist()

        # takes list of raw input variables
        norm = [float(i) / sum(raw_aa) for i in raw_aa]

        # random sequence generation (gen 0)
        seq = ""
        list_seq = []
        for i in range(int(length_aa)):
            get_weighted_aa = np.random.choice(list_aa_letters, 1, p=norm)
            list_seq.extend(get_weighted_aa)
        seq = "".join(list_seq)
        seq_parts_list.append(seq)
    return seq_parts_list


def seq_splitter(list_parts):
    splitted_entry_list = []
    for entries in list_parts:
        list_splitted_parts = [list(parts) for parts in entries]
        splitted_entry_list.append(list_splitted_parts)
    return splitted_entry_list


def seq_agglomerater(list_parts, part_tags: list):
    list_entries_agglomerated = []
    for entries in list_parts:
        list_agglomerated_parts = []
        for parts in entries:
            list_to_str = "".join(parts)
            list_agglomerated_parts.append(list_to_str)
        list_entries_agglomerated.append(list_agglomerated_parts)
    df = pd.DataFrame(list_entries_agglomerated, columns=part_tags)
    return df


@timingmethod
def aa_tree_pred(offspring_test_df: pd.DataFrame, sf_split: list):

    # imports for prediction
    non_sub_ccp_feat = pd.read_excel("")
    non_sub_df = pd.read_excel("")
    non_sub_labels = non_sub_df["labels"]

    # Create feature matrix
    sf_train = aa.SequenceFeature()
    df_parts_train = sf_train.get_df_parts(df_seq=non_sub_df)
    train_set = sf_train.feature_matrix(features=non_sub_ccp_feat["feature"], df_parts=df_parts_train)

    # fit the tree model
    tree = aa.TreeModel()
    tree = tree.fit(train_set, labels=non_sub_labels)

    # prediction of generated offspring
    sf_test = aa.SequenceFeature()
    df_parts_test = sf_test.get_df_parts(df_seq=offspring_test_df)
    test_set = sf_test.feature_matrix(features=non_sub_ccp_feat["feature"], df_parts=df_parts_test)
    pred, pred_std = tree.predict_proba(test_set)

    offspring_test_df["prediction"] = pred
    offspring_test_df["pred_std"] = pred_std

    return offspring_test_df

@timingmethod
def main_evo():

    toolbox = base.Toolbox()
    toolbox.register("gen_zero_maker", gen_zero_maker, "prop_SUB")
    pool_gen_0 = tools.initRepeat(list, toolbox.gen_zero_maker, 30)
    print(pool_gen_0)
    split_pool = seq_splitter(pool_gen_0)
    print(split_pool)
    agg_pool = seq_agglomerater(split_pool, ["jmd_n", "tmd", "jmd_c"])
    print(agg_pool)


# for debugging
# ______________________________________________________________________________________________________________________
if __name__ == "__main__":
    main_evo()
