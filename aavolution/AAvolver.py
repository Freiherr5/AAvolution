# general util import
import pandas as pd
from StandardConfig import timingmethod
import StandardConfig as stdc
import numpy as np
import random
import math
# ML tools import
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import aaanalysis as aa
# visualize
import matplotlib.pyplot as plt
import seaborn as sns


def proba_decision(float_proba):
    decimal, int_proba = math.modf(float_proba)
    if random.random() < decimal:  # decides if this instance chance occurs or not
        int_proba += 1
    return int(int_proba)


def seq_splitter(df_parts, part_tags: list):
    list_parts = df_parts[part_tags].to_numpy().tolist()
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

def evolution_display(list_pred_dfs):
    maxFitnessValues = []                         # WIP
    meanFitnessValues = []
    # for loop to unpack list of dataframes with results

    # add threshold lines of substrate performance
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='#BF2C34', label="max fitness")
    plt.plot(meanFitnessValues, color='#43ASBE', label="mean fitness")
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.legend()
    plt.show()

class AAvolutionizer:
    # script and module path! + seperator depending on operating system
    # __________________________________________________________________________________________________________________
    path_script, path_module, sep = stdc.find_folderpath()

    # import necessary tables as DataFrame
    # __________________________________________________________________________________________________________________
    path_data_sheets = f"{path_module.split('aavolution')[0]}_data{sep}"
    aa_propensities_df = pd.read_excel(f"{path_data_sheets}aa_propensities.xlsx").set_index("AA")
    length_dist_df = pd.read_excel(f"{path_data_sheets}length_dist.xlsx").set_index("len_tmd")

    # Evolutionary Algorithm Params
    # __________________________________________________________________________________________________________________
    POPULATION_SIZE = 200
    # these all per 100 AA or entries
    N_CROSSOVER = 50
    N_POINT_MUTATION = 5
    N_INDELS = 50
    MAX_GENERATIONS = 500
    MAX_TMD_LEN = 30
    MIN_TMD_LEN = 18

    @classmethod
    def set_mut_params(cls,
                       set_population_size: int = None,
                       n_crossover_per_seg: (int, float) = None,
                       n_point_mut: (int, float) = None,
                       n_indels: (int, float) = None,
                       max_gen: (int, float) = None,
                       max_len_tmd: (int, float) = None,
                       min_len_tmd: (int, float) = None, ):
        # population
        if isinstance(set_population_size, int) and set_population_size > 0:
            cls.POPULATION_SIZE = set_population_size
        # mutation
        if isinstance(n_crossover_per_seg, (int, float)) and n_crossover_per_seg >= 0:
            cls.N_CROSSOVER_SEGS = n_crossover_per_seg
        if isinstance(n_point_mut, (int, float)) and n_point_mut >= 0:
            cls.N_POINT_MUTATION = n_point_mut
        if isinstance(n_indels, (int, float)) and n_indels >= 0:
            cls.N_INDELS = n_indels
        # threshold params
        if isinstance(max_gen, int) and max_gen > 0:
            cls.MAX_GENERATIONS = max_gen
        if isinstance(max_len_tmd, int) and max_len_tmd > 0:
            cls.MAX_TMD_LEN = max_len_tmd
        if isinstance(min_len_tmd, int) and min_len_tmd > 0:
            cls.MIN_TMD_LEN = min_len_tmd

        print(f"""
                Current parameter settings
                ==========================
                
                population: {cls.POPULATION_SIZE}
                max generations: {cls.MAX_GENERATIONS}
                max length of the TMD: {cls.MAX_TMD_LEN}
                min length of the TMD: {cls.MIN_TMD_LEN}
                
                average mutagenic events for the population 
                (input values are per 100 individuals)
                ___________________________________________
                frequency crossover within each segment: {(cls.N_CROSSOVER * cls.POPULATION_SIZE) / 100} / {cls.POPULATION_SIZE}
                frequency point mutations: {cls.N_POINT_MUTATION} / 100 amino acids
                frequency indels: {cls.N_INDELS} / 100 amino acids (TMD only)
                """)

    # INITIALIZING
    # __________________________________________________________________________________________________________________
    # connect mode with correct table entries: aa_propensity, length_dist
    # side note: prop means propensity
    dict_mode = {"rand_prop": {"jmd_n": "codon_table",
                               "tmd": "codon_table",
                               "jmd_c": "codon_table"},
                 "prop_N_out": {"jmd_n": "N_out_JMD_N",
                                "tmd": "N_out_TMD",
                                "jmd_c": "N_out_JMD_C"},
                 "prop_SUB": {"jmd_n": "SUB_JMD_N",
                              "tmd": "SUB_TMD",
                              "jmd_c": "SUB_JMD_C"}}


    def __init__(self, mode, set_part_slices, df_seq_parts):
        self.mode = mode
        self.set_part_slices = set_part_slices
        self.df_seq_parts = df_seq_parts


    @classmethod
    def gen_zero_maker(cls, mode: str = "rand_prop", **parts_kwargs):

        # check inputs
        # ______________________________________________________________________________________________________________
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
            raise ValueError(f"parts list ({len(set_part_slices)}) and parts length list "
                             f"({len(set_len_part_slices)}) are not of same length!")

        if mode not in cls.dict_mode.keys():
            mode = "rand_prop"

        # generate the population
        n = 0
        list_seq_all_parts = []
        while n < cls.POPULATION_SIZE:
            seq_parts_list = []
            for part_seq, len_part in zip(set_part_slices, set_len_part_slices):

                raw_aa = cls.aa_propensities_df[cls.dict_mode[mode][part_seq]]
                # properties of length
                if mode == "prop_N_out":
                    if len_part < 1:
                        length_aa_series = cls.length_dist_df[cls.dict_mode[mode][part_seq]].dropna()
                        pre_p = length_aa_series.to_numpy().tolist()
                        norm_p = [p / sum(pre_p) for p in pre_p]
                        length_aa = np.random.choice(length_aa_series.index.tolist(), 1, p=norm_p)
                    else:
                        length_aa = len_part
                elif mode == "prop_SUB":
                    if len_part < 1:
                        length_aa_series = cls.length_dist_df[cls.dict_mode[mode][part_seq]].dropna()
                        pre_p = length_aa_series.to_numpy().tolist()
                        norm_p = [p / sum(pre_p) for p in pre_p]
                        length_aa = np.random.choice(length_aa_series.index.tolist(), 1, p=norm_p)
                    else:
                        length_aa = len_part
                else:
                    if len_part < 1:
                        length_aa = np.random.randint(15, 35)  # based on TMHMM arbitrary rules
                    else:
                        length_aa = len_part

                list_aa_letters = cls.aa_propensities_df.index.tolist()

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
            list_seq_all_parts.append(seq_parts_list)
            n += 1
        df_seq_parts = pd.DataFrame(list_seq_all_parts, columns=set_part_slices)
        AAvolutionizer._get_parts(mode, set_part_slices)
        return cls(mode, set_part_slices, df_seq_parts)


    def return_parts(self):
        return self.df_seq_parts

    # save the correct amino acid propensity scales for the mutations
    # __________________________________________________________________________________________________________________
    slices_propensity_data = None

    @classmethod
    def _get_parts(cls, mode, set_part_slices):
        list_propensity_data = []
        for part_seq in set_part_slices:
            path_script, path_module, sep = stdc.find_folderpath()
            path_data_sheets = f"{path_module.split('aavolution')[0]}_data{sep}"
            aa_propensities_df = pd.read_excel(f"{path_data_sheets}aa_propensities.xlsx").set_index("AA")

            aa_list = aa_propensities_df[AAvolutionizer.dict_mode[mode][part_seq]].dropna()
            pre_p = aa_list.to_numpy().tolist()
            norm = [p / sum(pre_p) for p in pre_p]
            list_aa_letters = aa_list.index.tolist()
            list_propensity_data.append([list_aa_letters, norm])
        cls.slices_propensity_data = list_propensity_data

    # MUTAGENESIS
    # __________________________________________________________________________________________________________________
    @staticmethod
    def single_point_mutation(split_aa_list):
        mut_proba = (AAvolutionizer.N_POINT_MUTATION / 100)  # proba per 100 AA
        split_aa_list_mut = []
        for entry in split_aa_list:
            list_allels_entry = []
            for allels in entry:
                mut_proba_allel = proba_decision(mut_proba * len(allels))
                i = 0
                while i < mut_proba_allel:
                    pos = random.randint(0, len(allels)-1)
                    ind_allel = entry.index(allels)
                    list_aa_letters = AAvolutionizer.slices_propensity_data[ind_allel][0]
                    norm = AAvolutionizer.slices_propensity_data[ind_allel][1]
                    get_weighted_aa = np.random.choice(list_aa_letters, 1, p=norm)[0]
                    allels[pos] = get_weighted_aa
                    i += 1
                list_allels_entry.append(allels)
            split_aa_list_mut.append(list_allels_entry)
        return split_aa_list_mut

    @staticmethod
    def crossover_allel(split_aa_list):
        # for crossovers to work, split_aa_list must be changed while being iterrated
        counter = 0   # index current
        cross_proba = ((AAvolutionizer.N_CROSSOVER * AAvolutionizer.POPULATION_SIZE) / (100 * len(split_aa_list[0])))
        for entry in split_aa_list:
            for allels in entry:
                if random.random() < cross_proba:
                    # current entry
                    pos_start = random.randint(0, len(allels)-1)
                    pos_stop = random.randint(pos_start, len(allels))
                    seq_current = allels[pos_start:pos_stop]
                    ind_allel = entry.index(allels)           # index for both! allel index!

                    # get mating mate
                    while True:
                        index_random_mate = random.randint(0, len(split_aa_list)-1)
                        get_random_mate = split_aa_list[index_random_mate]
                        if get_random_mate != entry:
                            break
                    allel_random_mate = get_random_mate[ind_allel]
                    if len(allel_random_mate) > len(seq_current):
                        pos_start_rando = random.randint(0, len(allel_random_mate) - (1+len(seq_current)))
                        pos_stop_rando = pos_start_rando + len(seq_current)
                        seq_random_mate = allel_random_mate[pos_start_rando:pos_stop_rando]
                        # the CROSSOVER
                        allels[pos_start:pos_stop] = seq_random_mate
                        allel_random_mate[pos_start_rando:pos_stop_rando] = seq_current

                        # applying the crossover to the current split_aa_list, change in place while iterrating
                        split_aa_list[counter][ind_allel] = allels
                        split_aa_list[index_random_mate][ind_allel] = allel_random_mate
            counter += 1
        return split_aa_list


    def indels_mutation(self, split_aa_seq, indel_tmd=True, indel_jmd=False):
        indel_proba = (AAvolutionizer.N_INDELS / 100)  # proba per 100 AA

        # translate parts of allel into bool
        parts_bool_indel = []
        for parts in self.set_part_slices:
            if (parts == "jmd_n") or (parts == "jmd_c"):
                parts_bool_indel.append(indel_jmd)
            else:
                parts_bool_indel.append(indel_tmd)

        for entry in split_aa_seq:
            for allels, part_bool in zip(entry, parts_bool_indel):
                indel_proba_allel = proba_decision(indel_proba * len(allels))
                i = 0
                if part_bool:
                    while i < indel_proba_allel:
                        pos = random.randint(0, len(allels)-1)
                        ind_allel = entry.index(allels)
                        if random.random() < 0.5:  # insert
                            list_aa_letters = AAvolutionizer.slices_propensity_data[ind_allel][0]
                            norm = AAvolutionizer.slices_propensity_data[ind_allel][1]
                            get_weighted_aa = np.random.choice(list_aa_letters, 1, p=norm)[0]
                            allels.insert(pos, get_weighted_aa)
                        else:  # delete
                            del allels[pos]
                        i += 1
        return split_aa_seq

    # Filtering of TMD lengths
    # __________________________________________________________________________________________________________________
    @staticmethod
    def tmd_length_filter(df_aa_split):
        index_list_tmd = []
        count = 0
        for col in df_aa_split.columns.tolist():
            if col == "tmd":
                index_list_tmd.append(count)
            count += 1
        if len(index_list_tmd) == 0:
            raise ValueError("no tmd in DataFrame")
        for index, rows in df_aa_split.iterrows():
            for ind in index_list_tmd:
                if (len(rows.iloc[ind]) < AAvolutionizer.MIN_TMD_LEN) or (len(rows.iloc[ind]) > AAvolutionizer.MAX_TMD_LEN):
                    if index in df_aa_split.index.tolist():
                        df_aa_split = df_aa_split.drop(index, axis=0)
        return df_aa_split


    @staticmethod
    def aa_tree_pred(non_sub_ccp_feat: pd.DataFrame,
                     non_sub_df: pd.DataFrame,
                     offspring_test_df: pd.DataFrame,
                     sf_split: list):

        #  get labels
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

    def mate_survivors(self, split_aa_df):
        current_survivor_count = split_aa_df.shape[0]
        target_population = AAvolutionizer.POPULATION_SIZE

        allels_tags = self.set_part_slices

        list_parts_sorted = []
        for tag in allels_tags:
            allel_list = split_aa_df[tag].to_numpy().tolist()
            list_parts_sorted.append(allel_list)

        surviver_list = split_aa_df[allels_tags].to_numpy().tolist()
        count = 0
        while count < (target_population-current_survivor_count):
            child = []
            for allel_genes in list_parts_sorted:
                allel_gene = random.choice(allel_genes)
                child.append(allel_gene)
            surviver_list.append(child)
            count += 1
        return surviver_list

@timingmethod
def main_evo():
    import copy
    initialize = AAvolutionizer.gen_zero_maker("prop_SUB")
    pool_gen_0 = initialize.return_parts()
    print(pool_gen_0)
    split_pool = seq_splitter(pool_gen_0, ["jmd_n", "tmd", "jmd_c"])
    print(split_pool)
    split_pool_for_mut = copy.deepcopy(split_pool)
    mut_pool = AAvolutionizer.single_point_mutation(split_pool_for_mut)
    print(mut_pool)
    new_mut_list = copy.deepcopy(mut_pool)
    cross_pool = AAvolutionizer.crossover_allel(new_mut_list)
    print(cross_pool)
    new_cross_pool = copy.deepcopy(cross_pool)
    indel_pool = initialize.indels_mutation(new_cross_pool)
    print(indel_pool)
    print(split_pool == mut_pool)
    print(mut_pool == cross_pool)
    agg_pool = seq_agglomerater(indel_pool, ["jmd_n", "tmd", "jmd_c"])
    print(agg_pool)
    tmd_filter = initialize.tmd_length_filter(agg_pool)
    print(tmd_filter)
    new_gen = initialize.mate_survivors(tmd_filter)
    print(len(new_gen))
    print(new_gen)
# for debugging
# ______________________________________________________________________________________________________________________
if __name__ == "__main__":
    main_evo()
