import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from utils.print_outputs import *
from utils.utils import *
from scipy.stats import wilcoxon


class Eval_Trees:
    def __init__(self, test_dir, training_dir=None):
        self._test_dir = test_dir
        self._training_dir = training_dir
        if training_dir is not None:
            self._tree_dir = os.path.join(self._training_dir, "Trees_dir")
        self._trees = []
        self._scores = {}

    def load_trees(self, path):
        self._tree_dir = path
        for file in os.listdir(path):
            if file.endswith(".pickle"):
                tree = get_tree(os.path.join(path, file))
                self._trees.append(tree)

    def get_from_list_dir(self):
        if type(self._test_dir) == dict:
            for key in self._test_dir.keys():
                self._scores[key] = self.get_scores(self._test_dir[key])
        else:
            self._scores = self.get_scores(self._test_dir)

    def get_scores(self, path):
        log_dir = os.path.join(path, "eval_log", str(0))
        files = os.listdir(log_dir)
        scores = {}
        for file in files:
            if "completed" in file:
                key = "completed"
                dtype = int
            elif "n_kills" in file:
                key = "n_kills"
                dtype = int
            elif "rewards" in file:
                key = "rewards"
                dtype = float
            else:
                raise ValueError(f"Unknown file: {file}")
            file = os.path.join(log_dir, file)

            with open(file, "r") as f:
                array = f.read()

            array = array.strip("[").strip("]\n").split(", ")
            data = np.array(array, dtype=dtype)
            scores[key] = data
        return scores

    def plot_evals(self):
        self.get_from_list_dir()
        if type(self._test_dir) == dict:
            data = self._scores["coach"]["rewards"]
            sns.catplot(data=data, kind="box")
            path = os.path.join("logs/qd-marl/test", "plot.png")
            plt.savefig(path)


class Eval_MEs:
    def __init__(self, dir, data_dir = None, add_title=None, selection_type=None, experiment=None):
        # dir is the outer directory of the MEs
        # self._log_dir = dir

        self._global_dir = dir
        self._data_dir = data_dir
        self._mes = []
        self._me_dfs = []
        self._best_me = None
        self._means_me = None
        self._selection_type = selection_type
        self._experiment = experiment
        self._palette_list = self._palette_list = ["red", "blue", "green"]
        self._palette = {
            "random": "tomato",
            "best": "cornflowerblue",
            "coach": "mediumspringgreen",
        }
        self._order = ["coach", "best", "random"]
        self._add_title = add_title

    def load_mes(self):
        run_dirs = [
            f
            for f in os.listdir(self._log_dir)
            if os.path.isdir(os.path.join(self._log_dir, f))
        ]
        run_dirs = [os.path.join(self._log_dir, run, "ME") for run in run_dirs]
        for dir in run_dirs:
            me_files = os.listdir(dir)
            for file in me_files:
                if file.endswith(".pickle"):
                    me = get_me(os.path.join(dir, file))
                    self._mes.append(me)
                    self._me_dfs.append(me._archive.data(return_type="pandas"))
        max = -np.inf
        for df in self._me_dfs:
            maxes_mean = np.mean(
                df["objective"].sort_values(ascending=False).iloc[0:12]
            )
            if maxes_mean > max:
                max = maxes_mean
                self._best_me = df

    def get_means_for_cells(self):
        objectives = []
        df = pd.concat(self._me_dfs)
        df = df.drop(columns=["measures_0", "measures_1", "threshold", "tree"])
        grouped = df.groupby(["index", "solution_0", "solution_1"], as_index=False)
        means = grouped.mean()
        return means

    def get_max_for_cells(self):
        objectives = []
        df = pd.concat(self._me_dfs)
        df = df.drop(columns=["measures_0", "measures_1", "threshold", "tree"])
        grouped = df.groupby(["index", "solution_0", "solution_1"], as_index=False)
        maxs = grouped.max()
        return maxs

    def plot_best_me(self):
        self._best_me["solution_0"] = self._best_me["solution_0"].astype(int)
        self._best_me["solution_1"] = self._best_me["solution_1"].astype(int)
        self._best_me = self._best_me.pivot(
            index="solution_1", columns="solution_0", values="objective"
        )
        xticks = np.arange(0, 10, 1)
        yticks = np.arange(0, 10, 1)
        sns.heatmap(
            data=self._best_me,
            cmap="magma",
            xticklabels=xticks,
            yticklabels=yticks,
            vmin=-20,
            vmax=30,
        )
        plt.title("MAP-Elites archive: best fitness values over 10 runs")
        plt.xlabel("Tree's depth")
        plt.ylabel("Leaves' entropy bin")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.savefig(self._log_dir + f"/best_heatmap.png")
        plt.close()

    def plot_mean_heatmap(self):
        df_means = self.get_means_for_cells()
        df_means["solution_0"] = df_means["solution_0"].astype(int)
        df_means["solution_1"] = df_means["solution_1"].astype(int)
        df_means = df_means.pivot(
            index="solution_1", columns="solution_0", values="objective"
        )
        self._means_me = df_means
        sns.heatmap(data=df_means, cmap="magma", vmin=-20, vmax=30)
        plt.title("MAP-Elites archive: mean fitness values per cell over 10 runs")
        plt.xlabel("Tree's depth")
        plt.ylabel("Leaves' entropy bin")
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.savefig(self._log_dir + "/mean_heatmap.png")
        plt.close()

    def plot_mean_best_heatmap(self):
        df_means = self.get_means_for_cells()
        df_means["solution_0"] = df_means["solution_0"].astype(int)
        df_means["solution_1"] = df_means["solution_1"].astype(int)
        self._means_me = df_means.pivot(
            index="solution_1", columns="solution_0", values="objective"
        )
        injection_type = self._set_injection_type()
        
        heatmap_dir = os.path.join(self._data_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        
        # self._means_me.to_csv(
        #     heatmap_dir
        #     + f"/{injection_type}-{self._experiment}-{self._selection_type}-mean_heatmap.csv"
        # )
        
        heat_matrix = self._means_me.to_numpy()
        matrix_shape = heat_matrix.shape
        
        with open(heatmap_dir + f"/{injection_type}-{self._experiment}-{self._selection_type}-mean_heatmap.txt", "wb") as f:
            f.write(
                f"{'row' : <10}{'col': <10}{'value': <10}\n".encode()
            )
            for row in range(matrix_shape[0]):
                for col in range(matrix_shape[1]):
                    f.write(
                        f"{row: <10}{col: <10}{heat_matrix[row][col]: <10.4}\n".encode()
                    )
        
        self._best_me["solution_0"] = self._best_me["solution_0"].astype(int)
        self._best_me["solution_1"] = self._best_me["solution_1"].astype(int)
        self._best_me = self._best_me.pivot(
            index="solution_1", columns="solution_0", values="objective"
        )
        heat_matrix = self._best_me.to_numpy()
        matrix_shape = heat_matrix.shape
        
        with open(heatmap_dir + f"/{injection_type}-{self._experiment}-{self._selection_type}-best_heatmap.txt", "wb") as f:
            f.write(
                f"{'row' : <10}{'col': <10}{'value': <10}\n".encode()
            )
            for row in range(matrix_shape[0]):
                for col in range(matrix_shape[1]):
                    f.write(
                        f"{row: <10}{col: <10}{heat_matrix[row][col]: <10.4}\n".encode()
                    )

        # self._best_me.to_csv(
        #     heatmap_dir
        #     + f"/{injection_type}-{self._experiment}-{self._selection_type}-best_heatmap.csv"
        # )
        figure, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 6))
        experiment = self.experiment_title(self._experiment)
        figure.subplots_adjust(wspace=0.05)
        figure.suptitle(
            f"{experiment}-{self._selection_type}'s archives - {self._add_title}"
        )
        xticks = np.arange(0, 10, 1)
        yvalues = np.arange(0.8, 1.0, 0.02).round(2)
        ypos = np.arange(0, 10, 1)
        g1 = sns.heatmap(
            data=self._means_me, cmap="magma", cbar=False, vmin=-20, vmax=30, ax=ax1
        )
        g2 = sns.heatmap(
            data=self._best_me, cmap="magma", cbar=False, vmin=-20, vmax=30, ax=ax2
        )

        # figure.suptitle(f"{self._experiment}-{self._selection_type}'s archive: best and mean fitness values over 10 runs")
        ax1.set_ylabel("Leaves' entropy")
        ax1.set_xlabel("Tree's depth")
        ax1.set_xlim(0, 10)
        ax1.set_yticks(ypos, yvalues, rotation=0)
        ax1.set_ylim(0, 10)
        ax1.set_title("Average archive")
        ax2.set_xlabel("Tree's depth")
        ax2.set_ylabel("")
        ax2.set_yticks([])
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.set_title("Best archive")
        figure.colorbar(
            g1.get_children()[0],
            ax=[ax1, ax2],
            orientation="vertical",
            location="right",
        )

        plt.savefig(
            self._log_dir
            + f"/{injection_type}-{self._experiment}-{self._selection_type}-me_heatmaps.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_every_me(self):

        figure = plt.figure(figsize=(20, 30))
        experiment = self.experiment_title(self._experiment)
        figure.suptitle(
            f"{experiment}-{self._selection_type}'s archives - {self._add_title}"
        )
        xticks = np.arange(0, 10, 1)
        yvalues = np.arange(0.8, 1.0, 0.02).round(2)
        ypos = np.arange(0, 10, 1)

        plot_counter = 1
        print_debugging(len(self._me_dfs))
        for df in self._me_dfs:
            if self._experiment == "me_fully_coevolutionary":
                plt.subplot(15, 8, plot_counter)
            else:
                plt.subplot(5, 2, plot_counter)
            df["solution_0"] = df["solution_0"].astype(int)
            df["solution_1"] = df["solution_1"].astype(int)
            df = df.pivot(index="solution_1", columns="solution_0", values="objective")
            sns.heatmap(data=df, cmap="magma", vmin=-20, vmax=30, cbar=False)
            plt.ylabel("Leaves' entropy")
            plt.xlabel("Tree's depth")
            plt.xlim(0, 10)
            plt.ylim(0, 10)
            plt.yticks(ypos, yvalues, rotation=0)

            plot_counter += 1
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        injection_type = "no_inj"
        if self._add_title == "with injection":
            injection_type = "inj"
        elif self._add_title == "no injection":
            injection_type = "no_inj"
        else:
            injection_type = "no_inj"
        plt.savefig(
            self._log_dir
            + f"/{injection_type}-{self._experiment}-{self._selection_type}-every_heatmaps.png",
            bbox_inches="tight",
        )
        plt.close()

    def plot_me_all_experiments(self):
        experiments = os.listdir(self._global_dir)
        for experiment in experiments:
            experiment_path = os.path.join(self._global_dir, experiment)
            if not os.path.isdir(experiment_path):
                continue
            else:
                selection_types = os.listdir(os.path.join(self._global_dir, experiment))
                for selection_type in selection_types:
                    selection_type_path = os.path.join(
                        self._global_dir, experiment, selection_type
                    )
                    if not os.path.isdir(selection_type_path):
                        continue
                    else:

                        self._selection_type = selection_type
                        self._experiment = experiment
                        self._log_dir = os.path.join(
                            self._global_dir,
                            experiment,
                            selection_type,
                            "magent_battlefield",
                        )
                        self.load_mes()
                        try:
                            print_info(f"Plotting {experiment}-{selection_type}")
                            self.plot_mean_best_heatmap()
                            self.plot_every_me()
                        except:
                            print_error(f"Error in {experiment}-{selection_type}")
                        self._me_dfs = []
                        self._mes = []

    def _set_injection_type(self):
        injection_type = "no_inj"
        if self._add_title == "with injection":
            injection_type = "inj"
        elif self._add_title == "no injection":
            injection_type = "no_inj"
        else:
            injection_type = "no_inj"
        return injection_type
    
    # def wilcoxon_from_dict(self, data_dict): #TODO
        

    def coverage_score(self):
        self._mes = []
        self._me_dfs = []
        experiments = os.listdir(self._global_dir)
        coverage_dict = {}
        for experiment in experiments:
            experiment_path = os.path.join(self._global_dir, experiment)
            if not os.path.isdir(experiment_path):
                continue
            else:
                coverage_dict[experiment] = {}
                selection_types = os.listdir(os.path.join(self._global_dir, experiment))
                for selection_type in selection_types:
                    selection_type_path = os.path.join(
                        self._global_dir, experiment, selection_type
                    )
                    if not os.path.isdir(selection_type_path):
                        continue
                    else:
                        coverage_dict[experiment][selection_type] = []
                        path = os.path.join(
                            self._global_dir,
                            experiment,
                            selection_type,
                            "magent_battlefield",
                        )
                        runs = os.listdir(path)
                        for run in runs:
                            run_path = os.path.join(path, run)
                            if not os.path.isdir(run_path):
                                continue
                            else:
                                run_path = os.path.join(path, run, "ME")
                                me_files = os.listdir(run_path)
                                for file in me_files:
                                    if file.endswith(".pickle"):
                                        me = get_me(os.path.join(run_path, file))
                                        me_df = me._archive.data(return_type="pandas")
                                        coverage = len(me_df) / (
                                            me._map_size[0] * me._map_size[1]
                                        )
                                        coverage_dict[experiment][
                                            selection_type
                                        ].append(coverage)
        figure, ax = plt.subplots(1, 2, figsize=(15, 6))
        figure.subplots_adjust(wspace=0.05)
        first_plot = True
        injection_type = self._set_injection_type()
        
        coverage_dir = os.path.join(self._data_dir, "coverage")
        os.makedirs(coverage_dir, exist_ok=True)
        
        for experiment in coverage_dict.keys():
            for selection_type in coverage_dict[experiment].keys():
                with open(
                    coverage_dir
                    + f"/{injection_type}-{experiment}-{selection_type}-coverage.txt",
                    "wb",
                ) as f:
                    f.write("coverage\n".encode())
                    for elem in coverage_dict[experiment][selection_type]:
                        f.write(f"{elem}\n".encode())

        for experiment, ax in zip(coverage_dict.keys(), ax.flatten()):
            coverage_dict[experiment] = {
                k: coverage_dict[experiment][k] for k in self._order
            }
            sns.boxplot(data=coverage_dict[experiment], palette=self._palette, ax=ax)
            ax.set_title(f"{self.experiment_title(experiment)}", fontsize=14)
            ax.set_xlabel("Selection type", fontsize=14)
            ax.tick_params(axis="both", which="major", labelsize=14)

            if first_plot:
                first_plot = False
                ax.set_ylabel("coverage", fontsize=14)
                ax.set_yticks(np.arange(0.75, 1.0, 0.05))
            else:
                ax.set_ylabel("")
                ypos = np.arange(0.75, 1.0, 0.05)
                yvalues = []
                ax.set_yticks(ypos, yvalues, rotation=0)
        figure.suptitle(f"Archive coverage - {self._add_title}", fontsize=16)
        plt.tight_layout()

        plt.savefig(self._global_dir + f"/{injection_type}-coverage_comparison.png")
        plt.close()

    def get_perfect_me(self):
        self._mes = []
        self._me_dfs = []
        experiments = os.listdir(self._global_dir)
        for experiment in experiments:
            experiment_path = os.path.join(self._global_dir, experiment)
            if not os.path.isdir(experiment_path):
                continue
            else:
                selection_types = os.listdir(os.path.join(self._global_dir, experiment))
                for selection_type in selection_types:
                    selection_type_path = os.path.join(
                        self._global_dir, experiment, selection_type
                    )
                    if not os.path.isdir(selection_type_path):
                        continue
                    else:
                        path = os.path.join(
                            self._global_dir,
                            experiment,
                            selection_type,
                            "magent_battlefield",
                        )
                        runs = os.listdir(path)
                        for run in runs:
                            run_path = os.path.join(path, run)
                            if not os.path.isdir(run_path):
                                continue
                            else:
                                run_path = os.path.join(path, run, "ME")
                                me_files = os.listdir(run_path)
                                for file in me_files:
                                    if file.endswith(".pickle"):
                                        me = get_me(os.path.join(run_path, file))
                                        self._me_dfs.append(
                                            me._archive.data(return_type="pandas")
                                        )
        maxs = self.get_max_for_cells()
        return maxs

    def mae_score(self):
        maxs = self.get_perfect_me()
        self._mes = []
        self._me_dfs = []
        experiments = os.listdir(self._global_dir)
        mae_dict = {}
        for experiment in experiments:
            experiment_path = os.path.join(self._global_dir, experiment)
            if not os.path.isdir(experiment_path):
                continue
            else:
                mae_dict[experiment] = {}
                selection_types = os.listdir(os.path.join(self._global_dir, experiment))
                for selection_type in selection_types:
                    selection_type_path = os.path.join(
                        self._global_dir, experiment, selection_type
                    )
                    if not os.path.isdir(selection_type_path):
                        continue
                    else:
                        mae_dict[experiment][selection_type] = []
                        path = os.path.join(
                            self._global_dir,
                            experiment,
                            selection_type,
                            "magent_battlefield",
                        )
                        runs = os.listdir(path)
                        for run in runs:
                            run_path = os.path.join(path, run)
                            if not os.path.isdir(run_path):
                                continue
                            else:
                                run_path = os.path.join(path, run, "ME")
                                me_files = os.listdir(run_path)
                                for file in me_files:
                                    mae = []
                                    if file.endswith(".pickle"):
                                        me = get_me(os.path.join(run_path, file))
                                        me_df = me._archive.data(return_type="pandas")
                                        for row in me_df.iterrows():
                                            index = row[1]["index"]
                                            maxs_fit = maxs[maxs["index"] == index][
                                                "objective"
                                            ].values[0]
                                            mae.append(
                                                abs(row[1]["objective"] - maxs_fit)
                                            )

                                        mae_ = np.mean(mae)
                                        mae_dict[experiment][selection_type].append(
                                            mae_
                                        )
        figure, ax = plt.subplots(1, 2, figsize=(15, 6))
        figure.subplots_adjust(wspace=0.05)
        first_plot = True
        for experiment, ax in zip(mae_dict.keys(), ax.flatten()):
            with open(self._global_dir + f"/{experiment}_rmse_data.txt", "wb") as f:
                f.write(str(mae_dict[experiment]).encode())
            sns.boxplot(data=mae_dict[experiment], palette=self._palette, ax=ax)
            ax.set_title(f"{self.experiment_title(experiment)}", fontsize=14)
            ax.set_xlabel("Selection type", fontsize=14)
            ax.tick_params(axis="both", which="major", labelsize=14)

            if first_plot:
                first_plot = False
                ax.set_ylabel("MAE", fontsize=14)
                ax.set_yticks(np.arange(0, 40, 4))
            else:
                ypos = np.arange(0, 40, 4)
                yvalues = []
                ax.set_yticks(ypos, yvalues, rotation=0)
                ax.set_ylabel("")

        figure.suptitle(f"Archive MAE over the best fitness cell", fontsize=16)
        plt.tight_layout()
        injection_type = "no_inj"
        if self._add_title == "with injection":
            injection_type = "inj"
        elif self._add_title == "no injection":
            injection_type = "no_inj"
        else:
            injection_type = "no_inj"
        plt.savefig(self._global_dir + f"/mae_comparison.png")
        plt.close()

    def experiment_title(self, experiment):
        if experiment == "me-fully_coevolutionary":
            title = "Fully Coevolutionary"
        elif experiment == "me-single_me":
            title = "Singular MAP-Elites"
        else:
            title = experiment
        return title

    def mse_score(self):
        maxs = self.get_perfect_me()
        self._mes = []
        self._me_dfs = []
        experiments = os.listdir(self._global_dir)
        rmse_dict = {}
        for experiment in experiments:
            experiment_path = os.path.join(self._global_dir, experiment)
            if not os.path.isdir(experiment_path):
                continue
            else:
                rmse_dict[experiment] = {}
                selection_types = os.listdir(os.path.join(self._global_dir, experiment))
                for selection_type in selection_types:
                    selection_type_path = os.path.join(
                        self._global_dir, experiment, selection_type
                    )
                    if not os.path.isdir(selection_type_path):
                        continue
                    else:
                        rmse_dict[experiment][selection_type] = []
                        path = os.path.join(
                            self._global_dir,
                            experiment,
                            selection_type,
                            "magent_battlefield",
                        )
                        runs = os.listdir(path)
                        for run in runs:
                            run_path = os.path.join(path, run)
                            if not os.path.isdir(run_path):
                                continue
                            else:
                                run_path = os.path.join(path, run, "ME")
                                me_files = os.listdir(run_path)
                                for file in me_files:
                                    rmse = []
                                    if file.endswith(".pickle"):
                                        me = get_me(os.path.join(run_path, file))
                                        me_df = me._archive.data(return_type="pandas")
                                        for row in me_df.iterrows():
                                            index = row[1]["index"]
                                            maxs_fit = maxs[maxs["index"] == index][
                                                "objective"
                                            ].values[0]
                                            rmse.append(
                                                (row[1]["objective"] - maxs_fit) ** 2
                                            )

                                        rmse_ = np.sqrt(np.mean(rmse))
                                        rmse_dict[experiment][selection_type].append(
                                            rmse_
                                        )
        figure, ax = plt.subplots(1, 2, figsize=(15, 6))
        figure.subplots_adjust(wspace=0.05)
        first_plot = True
        injection_type = self._set_injection_type()
        
        rmse_dir = os.path.join(self._data_dir, "rmse")
        os.makedirs(rmse_dir, exist_ok=True)
        # save means
        for experiment in rmse_dict.keys():
            for selection_type in rmse_dict[experiment].keys():
                with open(
                    rmse_dir
                    + f"/{injection_type}-{experiment}-{selection_type}-rmse.txt",
                    "wb",
                ) as f:
                    f.write("rmse\n".encode())
                    for elem in rmse_dict[experiment][selection_type]:
                        f.write(f"{elem}\n".encode())

        # plot the RMSE for each experiment
        for experiment, ax in zip(rmse_dict.keys(), ax.flatten()):
            rmse_dict[experiment] = {k: rmse_dict[experiment][k] for k in self._order}
            sns.boxplot(data=rmse_dict[experiment], palette=self._palette, ax=ax)
            ax.set_title(f"{self.experiment_title(experiment)}", fontsize=14)
            ax.set_xlabel("Selection type", fontsize=14)
            ax.tick_params(axis="both", which="major", labelsize=14)

            if first_plot:
                first_plot = False
                ax.set_ylabel("RMSE", fontsize=14)
                ax.set_yticks(np.arange(0, 40, 4))
            else:
                ypos = np.arange(0, 40, 4)
                yvalues = []
                ax.set_yticks(ypos, yvalues, rotation=0)
                ax.set_ylabel("")
        figure.suptitle(
            f"Archive RMSE over the best fitness cell - {self._add_title}", fontsize=16
        )
        plt.tight_layout()
        plt.savefig(self._global_dir + f"/{injection_type}-mse_comparison.png")
        plt.close()


class Eval_Fitnesses:
    def __init__(self, path, data_dir = None, add_title=None):
        self._global_dir = path
        self._data_dir = data_dir
        self._data = []
        self._palette_list = self._palette_list = ["red", "blue", "green"]
        self._palette = {
            "random": "tomato",
            "best": "cornflowerblue",
            "coach": "mediumspringgreen",
        }
        self._save_path = path
        self._experiment = None
        self._last_gen = {}
        self._order = ["coach", "best", "random"]
        self._add_title = add_title

        self._fully_with_injection_mean = 36.967
        self._fully_with_injection_std = 10.002
        self._fully_without_injection_mean = 32.129
        self._fully_without_injection_std = 5.781

    def plot_all(self):
        experiments = os.listdir(self._global_dir)
        for experiment in experiments:
            esperiment_path = os.path.join(self._global_dir, experiment)
            if not os.path.isdir(esperiment_path):
                continue
            else:
                self._experiment = experiment
                self._last_gen[self._experiment] = {}
                selection_types = os.listdir(os.path.join(self._global_dir, experiment))
                for selection_type in selection_types:
                    selection_type_path = os.path.join(
                        self._global_dir, experiment, selection_type
                    )
                    if not os.path.isdir(selection_type_path):
                        continue
                    else:
                        path = os.path.join(
                            self._global_dir,
                            experiment,
                            selection_type,
                            "magent_battlefield",
                        )
                        self._last_gen[self._experiment][selection_type] = []
                        self._data.append(self.get_data(path, selection_type))
            self.plot_runs()
            self._data = []
        self.plot_last_gen()

    def get_data_from_agent_log(self, path, type_=""):

        subdir = os.listdir(path)
        maxs = []
        for folder in subdir:
            agent_dir = os.path.join(path, folder, "Evolution_dir", "Agents")
            if not os.path.exists(agent_dir):
                continue
            files = os.listdir(agent_dir)
            files = [os.path.join(agent_dir, file) for file in files]
            for file in files:
                if os.path.exists(file):

                    # self.replace_spaces_with_commas(file, file)

                    df = pd.read_csv(file)
                    max = df["Max"]
                    max = np.array(max)
                    maxs.append(max)
        min_len = np.min([len(max) for max in maxs])
        if min_len >= 40:
            min_len = 40
        data_per_gen = []
        for i in range(min_len):
            cur = []
            for j in range(len(maxs)):
                cur.append(maxs[j][i])
            data_per_gen.append(cur)

        self._last_gen[self._experiment][type_] = data_per_gen[-1]

        gen = np.arange(min_len)
        mean_per_gen = [np.mean(data) for data in data_per_gen]
        min_per_gen = [np.min(data) for data in data_per_gen]
        max_per_gen = [np.max(data) for data in data_per_gen]
        std_per_gen = [np.std(data) for data in data_per_gen]

        df = pd.DataFrame()
        df["Gen"] = gen
        df["Mean"] = mean_per_gen
        df["Min"] = min_per_gen
        df["Max"] = max_per_gen
        df["Std"] = std_per_gen
        df["-Std"] = df["Mean"] - df["Std"]
        df["+Std"] = df["Mean"] + df["Std"]
        df["type"] = type_

        return df

    def get_data(self, path, type_=""):

        subdir = os.listdir(path)
        maxs = []
        for folder in subdir:
            file = os.path.join(path, folder, "Evolution_dir", "bests.txt")
            if os.path.exists(file):
                df = pd.read_csv(file)
                max = df["Max"]
                max = np.array(max)
                maxs.append(max)

        min_len = np.min([len(max) for max in maxs])
        if min_len >= 40:
            min_len = 40
        data_per_gen = []
        for i in range(min_len):
            cur = []
            for j in range(len(maxs)):
                cur.append(maxs[j][i])
            data_per_gen.append(cur)

        self._last_gen[self._experiment][type_] = data_per_gen[-1]

        gen = np.arange(min_len)
        mean_per_gen = [np.mean(data) for data in data_per_gen]
        min_per_gen = [np.min(data) for data in data_per_gen]
        max_per_gen = [np.max(data) for data in data_per_gen]
        std_per_gen = [np.std(data) for data in data_per_gen]

        df = pd.DataFrame()
        df["Gen"] = gen
        df["Mean"] = mean_per_gen
        df["Min"] = min_per_gen
        df["Max"] = max_per_gen
        df["Std"] = std_per_gen
        df["-Std"] = df["Mean"] - df["Std"]
        df["+Std"] = df["Mean"] + df["Std"]
        df["type"] = type_

        return df

    def experiment_title(self, experiment):
        if experiment == "me-fully_coevolutionary":
            title = "Fully Coevolutionary"
        elif experiment == "me-single_me":
            title = "Singular MAP-Elites"
        else:
            title = experiment
        return title

    def _set_injection_type(self):
        injection_type = "no_inj"
        if self._add_title == "with injection":
            injection_type = "inj"
        elif self._add_title == "no injection":
            injection_type = "no_inj"
        else:
            injection_type = "no_inj"
        return injection_type

    def plot_runs(self):
        injection_type = self._set_injection_type()
        for i in range(len(self._data)):
            label = self._data[i]["type"].values[0]
            self._data[i].to_csv(
                f"{self._save_path}/{injection_type}-{self._experiment}-{label}-training_fitness.csv"
            )
            trend_fitness_dir = os.path.join(self._data_dir, "trend_fitness")
            os.makedirs(trend_fitness_dir, exist_ok=True)            
            
            with open(trend_fitness_dir + f"/{injection_type}-{self._experiment}-{label}-training_fitness.txt", "wb") as f:
                f.write(f"{'i' : <10} {'m': <10} {'s': <10}\n".encode())
                for index, row in self._data[i].iterrows():
                    f.write(f"{row['Gen']: <10} {row['Mean']: <10.2f} {row['Std']: <10.2f}\n".encode())
            
            if label == "random":
                palette = self._palette_list[0]
            elif label == "best":
                palette = self._palette_list[1]
            else:
                palette = self._palette_list[2]

            colors = sns.light_palette(palette, 8)
            sns.lineplot(
                data=self._data[i], x="Gen", y="Mean", label=label, color=colors[-1]
            )
            # sns.lineplot(data=self._data[i], x="Gen", y="-Std", color=colors[2])
            # sns.lineplot(data=self._data[i], x="Gen", y="+Std", color=colors[2])
            plt.fill_between(
                self._data[i]["Gen"],
                self._data[i]["-Std"],
                self._data[i]["+Std"],
                color=colors[2],
                alpha=0.2,
            )

        # Plot the GE results as thresholds
        # y = 40*[self._fully_with_injection_mean]
        # plt.plot(np.arange(0,40,1), y, color="black", linestyle="--", label="GE F-Coev inj.")
        # y = 40*[self._fully_without_injection_mean]
        # plt.plot(np.arange(0,40,1), y, color="orange", linestyle="--", label="GE F-Coev no inj.")
        experiment = self.experiment_title(self._experiment)
        plt.title(
            f"{experiment} - {self._add_title} \n Best Individual Fitness Achieved by Generation",
            fontsize=10,
        )
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.xticks(np.arange(0, len(self._data[i]["Gen"]), 2))
        plt.yticks(np.arange(0, 60, 10))
        plt.ylim(0, 60)
        plt.xlim(-1, 39)
        plt.legend()
        plt.grid(True)
        plt.savefig(
            f"{self._save_path}/{injection_type}-{self._experiment}_fitness.png"
        )
        plt.close()

    def plot_last_gen(self):
        figure, ax = plt.subplots(1, 2, figsize=(15, 6))
        figure.subplots_adjust(wspace=0.05)
        first_plot = True
        injection_type = self._set_injection_type()
        
        last_fitness_dir = os.path.join(self._data_dir, "last_fitness")
        os.makedirs(last_fitness_dir, exist_ok=True)
        
        for experiment in self._last_gen.keys():
            for selection_type in self._last_gen[experiment].keys():
                with open(
                    last_fitness_dir
                    + f"/{injection_type}-{experiment}-{selection_type}-last_fitness.txt",
                    "wb",
                ) as f:
                    f.write("fitness\n".encode())
                    for elem in self._last_gen[experiment][selection_type]:
                        f.write(f"{elem}\n".encode())

        for experiment, ax in zip(self._last_gen.keys(), ax.flatten()):
            with open(self._global_dir + f"/{experiment}_last_fitness.txt", "wb") as f:
                f.write(str(self._last_gen[experiment]).encode())
            self._last_gen[experiment] = {
                k: self._last_gen[experiment][k] for k in self._order
            }
            sns.boxplot(data=self._last_gen[experiment], palette=self._palette, ax=ax)
            ax.set_title(f"{self.experiment_title(experiment)}", fontsize=14)
            ax.set_xlabel("Selection type", fontsize=14)
            ax.tick_params(axis="both", which="major", labelsize=14)

            if first_plot:
                first_plot = False
                ax.set_ylabel("Fitness", fontsize=14)
                ax.set_yticks(np.arange(0, 60, 5))
            else:
                ypos = np.arange(0, 60, 5)
                yvalues = []
                ax.set_yticks(ypos, yvalues, rotation=0)
                ax.set_ylabel("")
        figure.suptitle(
            f"Best fitnesses yield per experiment - {self._add_title}", fontsize=14
        )
        plt.tight_layout()
        injection_type = "no_inj"
        if self._add_title == "with injection":
            injection_type = "inj"
        elif self._add_title == "no injection":
            injection_type = "no_inj"
        else:
            injection_type = "no_inj"
        plt.savefig(self._global_dir + f"/{injection_type}-final_results.png")
        plt.close()


class Eval_Test:
    def __init__(self, path, data_dir = None, add_title=None):
        self._global_dir = path
        self._data_dir = data_dir
        self._data = []
        self._palette = {
            "coach": "mediumspringgreen",
            "best": "cornflowerblue",
            "random": "tomato",
        }
        self._order = ["coach", "best", "random"]
        self._add_title = add_title

    def experiment_title(self, experiment):
        if experiment == "me-fully_coevolutionary":
            title = "Fully Coevolutionary"
        elif experiment == "me-single_me":
            title = "Singular MAP-Elites"
        else:
            title = experiment
        return title

    def get_data(self):

        experiments = os.listdir(self._global_dir)
        kill_dict = {}
        reward_dict = {}
        completed_dict = {}

        for experiment in experiments:
            experiment_path = os.path.join(self._global_dir, experiment)
            if not os.path.isdir(experiment_path):
                continue
            else:
                kill_dict[experiment] = {}
                reward_dict[experiment] = {}
                completed_dict[experiment] = {}
                selection_types = os.listdir(os.path.join(self._global_dir, experiment))
                for selection_type in selection_types:
                    selection_type_path = os.path.join(
                        self._global_dir, experiment, selection_type
                    )
                    if not os.path.isdir(selection_type_path):
                        continue
                    else:
                        kill_dict[experiment][selection_type] = []
                        reward_dict[experiment][selection_type] = []
                        completed_dict[experiment][selection_type] = []

                        path = os.path.join(
                            self._global_dir, experiment, selection_type
                        )
                        runs = os.listdir(path)
                        for run in runs:
                            run_path = os.path.join(path, run)
                            if not os.path.isdir(run_path):
                                continue
                            else:
                                eval_log = os.path.join(run_path, "eval_log", "0")
                                completed_path = os.path.join(
                                    eval_log, "log_completed.txt"
                                )
                                rewards_path = os.path.join(eval_log, "log_rewards.txt")
                                kills_path = os.path.join(eval_log, "log_kills.txt")

                                with open(completed_path, "r") as f:
                                    completed = (
                                        f.read().strip("[").strip("]\n").split(", ")
                                    )
                                f.close()
                                with open(rewards_path, "r") as f:
                                    rewards = (
                                        f.read().strip("[").strip("]\n").split(", ")
                                    )
                                f.close()
                                with open(kills_path, "r") as f:
                                    kills = f.read().strip("[").strip("]\n").split(", ")
                                f.close()

                                completed_dict[experiment][selection_type].append(
                                    int(completed[0])
                                )
                                reward_dict[experiment][selection_type].append(
                                    float(rewards[0])
                                )
                                kill_dict[experiment][selection_type].append(
                                    float(kills[0])
                                )
                        reward_dict[experiment][selection_type] = np.array(
                            reward_dict[experiment][selection_type]
                        ).round(2)
                        kill_dict[experiment][selection_type] = np.array(
                            kill_dict[experiment][selection_type]
                        ).round(2)

        print_info("Completed")
        print(completed_dict)
        print_info("Rewards")
        print(reward_dict)
        print_info("Kills")
        print(kill_dict)

        injection_type = self._set_injection_type()
        
        last_metrics_dir = os.path.join(self._data_dir, "last_metrics")
        os.makedirs(last_metrics_dir, exist_ok=True)

        for experiment in completed_dict.keys():
            for selection_type in completed_dict[experiment].keys():
                with open(
                    last_metrics_dir
                    + f"/{injection_type}-{experiment}-{selection_type}-final_metrics.txt",
                    "wb",
                ) as f:
                    f.write(
                        f"{'kills_stats' : <10} {'agents_scores_stats': <10} {'completed': <10}\n".encode()
                    )
                    for i in range(len(completed_dict[experiment][selection_type])):
                        f.write(
                            f"{kill_dict[experiment][selection_type][i]: <10} {reward_dict[experiment][selection_type][i]: <10} {completed_dict[experiment][selection_type][i]: <10}\n".encode()
                        )

        self.plot_dict_box(kill_dict, "Kills")
        self.plot_dict_box(reward_dict, "Reward")
        self.plot_dict_box(completed_dict, "Completed")

    def set_lims(self, metric):
        if "Kill" in metric:
            return 0, 14, 2
        elif "Reward" in metric:
            return -6, 15, 3
        elif "Completed" in metric:
            return 0, 100, 5
        else:
            return 0, 100, 5

    def _set_injection_type(self):
        injection_type = "no_inj"
        if self._add_title == "with injection":
            injection_type = "inj"
        elif self._add_title == "no injection":
            injection_type = "no_inj"
        else:
            injection_type = "no_inj"
        return injection_type

    def plot_dict_box(self, data_dict, metric):
        figure, ax = plt.subplots(1, 2, figsize=(15, 6))
        figure.subplots_adjust(wspace=0.05)
        first_plot = True

        ymin, ymax, step = self.set_lims(metric)
        # plot the RMSE for each experiment
        for experiment, ax in zip(data_dict.keys(), ax.flatten()):
            with open(self._global_dir + f"/{experiment}_plot_{metric}.txt", "wb") as f:
                f.write(str(data_dict[experiment]).encode())
            data_dict[experiment] = {k: data_dict[experiment][k] for k in self._order}
            sns.boxplot(data=data_dict[experiment], palette=self._palette, ax=ax)
            ax.set_title(f"{self.experiment_title(experiment)}", fontsize=14)
            ax.set_xlabel("Selection type", fontsize=14)
            ax.tick_params(axis="both", which="major", labelsize=14)

            if first_plot:
                first_plot = False
                ax.set_ylabel(f"{metric}", fontsize=14)
                ax.set_yticks(np.arange(ymin, ymax, step))
            else:
                ypos = np.arange(ymin, ymax, step)
                yvalues = []
                ax.set_yticks(ypos, yvalues, rotation=0)
                ax.set_ylabel("")
        figure.suptitle(f"{metric} of final teams - {self._add_title}", fontsize=16)
        plt.tight_layout()
        injection_type = self._set_injection_type()
        plt.savefig(self._global_dir + f"/{injection_type}-{metric}_final.png")
        plt.close()
        
class Eval_Wilcoxon():
    def __init__(self, data_dir) -> None:
        self._data_dir = data_dir
        
        save_dir = self._data_dir.replace(self._data_dir.split("/")[-1], "")
        
        self._save_dir = os.path.join(save_dir, "wilcoxon")
        os.makedirs(self._save_dir, exist_ok=True)
        self._metric = self._data_dir.split("/")[-1]
        self._data = {}
        
    def get_data_single_column(self):
        data_files = []
        files_name = os.listdir(self._data_dir)
        print(len(files_name))
        inj_types = set()
        exp_types = set()
        sel_types = set()
        
        for file_ in files_name:
            data_files.append(os.path.join(self._data_dir, file_))
            data_file = os.path.join(self._data_dir, file_)
            file_ = file_.split(".")[0]
            file_ = file_.split("-")
            injection_type = file_[0]
            inj_types.add(injection_type)
            experiment = file_[1]
            exp_types.add(experiment)
            selection_type = file_[2]
            sel_types.add(selection_type)
            
        for inj in inj_types:
            self._data[inj] = {}
            for exp in exp_types:
                self._data[inj][exp] = {}
                for sel in sel_types:
                    self._data[inj][exp][sel] = []
        
        for file_ in files_name:
            data_files.append(os.path.join(self._data_dir, file_))
            data_file = os.path.join(self._data_dir, file_)
            file_ = file_.split(".")[0]
            file_ = file_.split("-")
            injection_type = file_[0]
            experiment = file_[1]
            selection_type = file_[2]
            with open(data_file, "r") as f:
                data = f.readlines()
                for i in range(1, len(data)):                    
                    self._data[injection_type][experiment][selection_type].append(float(data[i]))
                    
    def get_data_multi_column(self):
        data_files = []
        files_name = os.listdir(self._data_dir)
        print(len(files_name))
        inj_types = set()
        exp_types = set()
        sel_types = set()
        
        first_file = os.path.join(self._data_dir, files_name[0])
        
        with open(first_file, 'r') as file:
            content = file.readlines()

        # Clean up the header line and content
        cleaned_content = [content[0].replace('\xa0', ' ').strip()] + [line.strip() for line in content[1:]]

        # Write the cleaned content back to a temporary file
        cleaned_file_path = first_file
        with open(cleaned_file_path, 'w') as cleaned_file:
            cleaned_file.write("\n".join(cleaned_content))
        
        columns = pd.read_csv(first_file, delim_whitespace=True)
        print(columns)
        
        
        
        for file_ in files_name:
            data_files.append(os.path.join(self._data_dir, file_))
            data_file = os.path.join(self._data_dir, file_)
            file_ = file_.split(".")[0]
            file_ = file_.split("-")
            injection_type = file_[0]
            inj_types.add(injection_type)
            experiment = file_[1]
            exp_types.add(experiment)
            selection_type = file_[2]
            sel_types.add(selection_type)
        
        for col in columns:
            self._data[col] = {}
            for inj in inj_types:
                self._data[col][inj] = {}
                for exp in exp_types:
                    self._data[col][inj][exp] = {}
                    for sel in sel_types:
                        self._data[col][inj][exp][sel] = []
        
        for file_ in files_name:
            data_files.append(os.path.join(self._data_dir, file_))
            data_file = os.path.join(self._data_dir, file_)
            file_ = file_.split(".")[0]
            file_ = file_.split("-")
            injection_type = file_[0]
            experiment = file_[1]
            selection_type = file_[2]
            
            with open(data_file, 'r') as file:
                content = file.readlines()

            # Clean up the header line and content
            cleaned_content = [content[0].replace('\xa0', ' ').strip()] + [line.strip() for line in content[1:]]

            # Write the cleaned content back to a temporary file
            cleaned_file_path = data_file
            with open(cleaned_file_path, 'w') as cleaned_file:
                cleaned_file.write("\n".join(cleaned_content))
            
            df = pd.read_csv(data_file, delim_whitespace=True)
            for col in df.columns:
                array = df[col].values
                self._data[col][injection_type][experiment][selection_type] = array
                print(self._data[col][injection_type][experiment][selection_type])
        
    def means_for_fully_coev(self, data_array):
        means = []
        for i in range(0, len(data_array), 12):
            mean = np.mean(data_array[i:i+12])
            means.append(mean)
        return means
    
    def set_labels(self, name):
        if name == "me_fully_coevolutionary":
            return "FC"
        elif name == "me_single_me":
            return "SM"
        elif name == "inj":
            return "I"
        elif name == "no_inj":
            return "NI"
        elif name == "random":
            return "R"
        elif name == "best":
            return "B"
        elif name == "coach":
            return "C"
        else:
            return name
    
    def wilcoxon_from_dict(self, data_dict):
        inj_types = list(data_dict.keys())
        experiment_types = list(data_dict[inj_types[0]].keys())
        selection_types = list(data_dict[inj_types[0]][experiment_types[0]].keys())      
        wilcoxon_matrix = np.zeros((len(inj_types)*len(experiment_types)*len(selection_types), len(inj_types)*len(experiment_types)*len(selection_types)))  
        p_value_matrix = np.zeros((len(inj_types)*len(experiment_types)*len(selection_types), len(inj_types)*len(experiment_types)*len(selection_types)))
        
        labels = []
        
        for i in range(len(inj_types)):            
            for e in range(len(experiment_types)):                
                for s in range(len(selection_types)):
                    current = data_dict[inj_types[i]][experiment_types[e]][selection_types[s]]
                    current = np.array(current)
                    labels.append(f"{self.set_labels(inj_types[i])}-{self.set_labels(experiment_types[e])}-{self.set_labels(selection_types[s])}")
                    if len(current)>10:
                        current = self.means_for_fully_coev(current)
                    for _i in range(len(inj_types)):            
                        for _e in range(len(experiment_types)):                
                            for _s in range(len(selection_types)):
                                if i == _i and e == _e and s == _s:
                                    score =  -np.inf, 0
                                else:
                                    against = data_dict[inj_types[_i]][experiment_types[_e]][selection_types[_s]]
                                    against = np.array(against)
                                    
                                    if len(against)>10:
                                        against = self.means_for_fully_coev(against)
                                    score = wilcoxon(current, against, alternative="greater")
                                wilcoxon_matrix[i*len(experiment_types)*len(selection_types) + e*len(selection_types) + s][_i*len(experiment_types)*len(selection_types) + _e*len(selection_types) + _s] = score[0]
                                p_value_matrix[i*len(experiment_types)*len(selection_types) + e*len(selection_types) + s][_i*len(experiment_types)*len(selection_types) + _e*len(selection_types) + _s] = score[1]
        
        df_wilcoxon = pd.DataFrame(wilcoxon_matrix, columns=labels, index=labels)
        df_p_value = pd.DataFrame(p_value_matrix, columns=labels, index=labels)
        return wilcoxon_matrix, p_value_matrix, df_wilcoxon, df_p_value
    
    def run_wilcoxon(self, multi = False):
        if multi:
            self.get_data_multi_column()
            for key in self._data.keys():
                print(f"Evaluating Wilcoxon for {key}")
                wilcoxon_matrix, p_value_matrix, df_wilcoxon, df_p_values = self.wilcoxon_from_dict(self._data[key])
                print("Wilcoxon Matrix")
                print(df_wilcoxon)
                print("P-Value Matrix")
                print(df_p_values)
                df_wilcoxon.to_csv(os.path.join(self._save_dir, f"{key}-wilcoxon_matrix.csv"))
                df_p_values.to_csv(os.path.join(self._save_dir, f"{key}-p_value_matrix.csv"))
        else:
            self.get_data_single_column()
            print("Evaluating Wilcoxon")
            wilcoxon_matrix, p_value_matrix, df_wilcoxon, df_p_values = self.wilcoxon_from_dict(self._data)
        
            print("Wilcoxon Matrix")
            print(df_wilcoxon)
            print("P-Value Matrix")
            print(df_p_values)
        
            df_wilcoxon.to_csv(os.path.join(self._save_dir, f"{self._metric}-wilcoxon_matrix.csv"))
            df_p_values.to_csv(os.path.join(self._save_dir, f"{self._metric}-p_value_matrix.csv"))

if __name__ == "__main__":

    #EVAL RUNS
    
    path = "logs/qd-marl/training/with_injection"
    add_to_title = "with injection"
    training_data_dir = "logs/qd-marl/training/data"
    # eval_me = Eval_MEs(path, training_data_dir, add_to_title)
    # eval_me.plot_me_all_experiments()
    # eval_me.coverage_score()
    # eval_me.mse_score()

    eval_fitness = Eval_Fitnesses(path, training_data_dir, add_to_title)
    eval_fitness.plot_all()

    # test_data_dir = "logs/qd-marl/test/data"
    # path = "logs/qd-marl/test/with_injection"
    # eval_test = Eval_Test(path, test_data_dir, add_to_title)
    # eval_test.get_data()

    path = "logs/qd-marl/training/without_injection"
    add_to_title = "without injection"
    # eval_me = Eval_MEs(path, training_data_dir, add_to_title)
    # eval_me.plot_me_all_experiments()
    # eval_me.coverage_score()
    # eval_me.mse_score()

    eval_fitness = Eval_Fitnesses(path, training_data_dir, add_to_title)
    eval_fitness.plot_all()

    # path = "logs/qd-marl/test/without_injection"
    # add_to_title = "without injection"
    # eval_test = Eval_Test(path, test_data_dir, add_to_title)
    # eval_test.get_data()

    #EVAL WILCOXON
    # data_dir = "logs/qd-marl/test/data/last_metrics"
    # eval_wilcoxon = Eval_Wilcoxon(data_dir)
    # eval_wilcoxon.run_wilcoxon(True)