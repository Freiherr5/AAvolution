# ML import
from Run_AAvolution import run_aavolution
# general util import
import pandas as pd
import numpy as np
# visualize
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

def benchmark_aavolution():


    path_data = "/home/freiherr/PycharmProjects/AAvolution/aavolution/w8_uniprot_refining"
    test_feat = pd.read_excel(f"{path_data}/CPP_feat_N8_SUB.xlsx")
    subexp_df = pd.read_excel(f"{path_data}/w8_TMD_refined.xlsx", "SUBEXP")
    sublit_df = pd.read_excel(f"{path_data}/w8_TMD_refined.xlsx", "SUBLIT")
    nonsub_df = pd.read_excel(f"{path_data}/w8_TMD_refined.xlsx", "NONSUB")

    # propensity
    #job_name = "AA-propensity dependent fitness"
    mode_list = ["prop_SUB", "prop_N_out", "rand_prop"]
    color_prop = ["#f0804e", "#c6417d", "#8707a6"]
    # population
    #job_name = "population-size dependent fitness"
    pop_list = [10, 20, 50, 100, 200, 500]
    color_pop = ["#f1f426", "#f0804e", "#c6417d", "#8707a6", "#110788", "#000000"]
    # point mutation
    #job_name = "point-mutation-rate dependent fitness"
    pmut_list = [0, 2, 4, 16, 64, 256]
    pmut_list_desc = ["0/100 AA", "2/100 AA", "4/100 AA", "16/100 AA", "64/100 AA", "256/100 AA"]
    color_pmut = ["#f1f426", "#f0804e", "#c6417d", "#8707a6", "#110788", "#000000"]
    # crossover segment intern
    #job_name = "crossover-rate dependent fitness"
    pseg_list = [0, 2, 4, 16, 64, 256]
    pseg_list_desc = ["0/200", "2/200", "4/200", "16/200", "64/200", "256/200"]
    color_pseg = ["#f1f426", "#f0804e", "#c6417d", "#8707a6", "#110788", "#000000"]
    # point mutation
    #job_name = "indel-rate (TMD) dependent fitness"
    ind_list = [0, 2, 4, 16, 64]
    ind_list_desc = ["0/100 AA", "2/100 AA", "4/100 AA", "16/100 AA", "64/100 AA", "256/100 AA"]
    color_ind = ["#f1f426", "#f0804e", "#c6417d", "#8707a6", "#110788"]

    mean_gens_list = []
    APP_value_list = []

    for mode, tag in zip(mode_list, mode_list):
        dict_evo_settings = {"set_population_size": 200,
                             "max_gen": 100,
                             "n_point_mut": 4,
                             "n_crossover_per_seg": 2,
                             "n_indels": 2}
        test_seq = pd.concat([subexp_df, sublit_df, nonsub_df], axis=0).reset_index().drop("index", axis=1)
        bench_seq = pd.concat([subexp_df, sublit_df], axis=0).reset_index().drop("index", axis=1)

        job_name, max_mean_gens, mean_bench, max_bench, APP_value = run_aavolution(job_name=job_name,
                                                                                   mode=mode,
                                                                                   parts=["jmd_n", "tmd", "jmd_c"],
                                                                                   df_seq_train=test_seq,
                                                                                   df_feat_train=test_feat,
                                                                                   df_bench_pred=bench_seq,
                                                                                   propensity_increment_display=1,
                                                                                   dict_evo_params=dict_evo_settings,
                                                                                   export_data=True)
        max_mean_gens["mean"] = max_mean_gens["mean"].div(max_bench)
        APP_value = APP_value/max_bench
        mean_gens_list.append(max_mean_gens.rename(columns={"mean": tag, "max": "max"})[tag])
        APP_value_list.append(APP_value)


    # for loop to unpack list of dataframes with results
    # add threshold lines of substrate performance
    pred_df = pd.concat(mean_gens_list, axis=1)
    APP_benchmark = np.mean(APP_value_list)

    sns.set_style("whitegrid")
    ax = pred_df.plot.line(color=color_prop)  # colors for max and mean fitness
    ax.spines[['right', 'top']].set_visible(False)

    # APP for comparison
    ax.text(pred_df.shape[0], APP_benchmark - 0.005, f"APP (pred: {round(APP_benchmark, 3)})", ha="right",
            va="top", color='#9f19d7', fontweight="heavy", fontsize=11,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.axhline(y=APP_benchmark, color='#9f19d7', linestyle='--',
               path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.set_xlabel('Generation', fontweight="bold")
    ax.set_ylabel('Max / Average Fitness', fontweight="bold")
    ax.set_title(f'{job_name}', fontweight="bold", fontsize=15)
    ax.legend()
    # ax.set_xticks(np.linspace(0, list_pred_dfs.shape[0]-1, list_pred_dfs.shape[0]))
    plt.savefig(f"_benchmark_AAvolution/{job_name}_evolution_progress.png", bbox_inches="tight", dpi=300)
    plt.show()


if __name__ == "__main__":
    pass
    # benchmark_aavolution()
