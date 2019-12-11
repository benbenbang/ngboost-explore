"""
Author: Ben CHEN
Github: benbenbang
Created Date: 2019-12-02
Last Modified by: benbenbang
Last Modified: 2019-12-10
"""

import os
import warnings
from timeit import default_timer

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
from libs.params import *
from libs.preprocessing import make_data
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.scores import MLE
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

warnings.filterwarnings("ignore")

num_rounds = [1000, 5000]
estimators = [1000, 5000]
lrates = [0.009, 0.007]

# Set the configs in the docker-compose file
seed_int = int(os.environ["SEED"])
num_plots = int(os.environ["NUM_PLOTS"])
dockerenv = os.path.isfile("/.dockerenv")

seed = np.random.seed(seed_int)
sns.set_style(style)


def normalize_ticks(ax):
    cln_ticks = lambda t: float(t.replace("−", "-"))
    get_ticks = lambda ax: list(
        map(cln_ticks, [item.get_text() for item in ax.get_yticklabels()])
    )
    labels = get_ticks(ax)
    labels = [f"{l / np.max(labels):.3f}" for l in labels]
    ax.set_yticklabels(labels)
    return ax


def gen_xspan(mean, std, n=4, num=100):
    """Generate span for x axis to plot the probabilty distribution

    Args:
        mean (float): The first parameter.
        std (float): The second parameter.
        n (int): The number used to calculate the approximate percentage of data that fall in 68, 95, 99.7
        num (int): number of points, the greater this number, the smoother the plot

    Returns:
        (np.ndarray): Span of X

    """
    return np.linspace(mean - n * std, mean + n * std, num)


def benchmark():
    """Main App of this benchmark"""
    with tqdm(total=9) as pbar:
        # Load Dataset / Initialize table
        pbar.set_description("Loading Dataset and init default table")
        X_train, X_test, y_train, y_test = make_data(test_size=0.2)

        table = PrettyTable()
        table.field_names = ["Name", "Iteration", "Estimators", "RMSE", "Time"]
        pbar.update()

        # Natural Gradient Boosting
        for est, lr in zip(estimators, lrates):
            pbar.set_description(
                f"Building NGBoost w/ {est} Estimators and learning rate = {lr}"
            )
            ngb = NGBRegressor(
                Base=default_tree_learner,
                Dist=Normal,
                Score=MLE,
                learning_rate=lr,
                natural_gradient=True,
                minibatch_frac=0.2,
                n_estimators=est,
                verbose=False,
            )
            start = default_timer()
            ngb = ngb.fit(X_train, y_train)
            time_ngb = default_timer() - start
            y_pred_ngb = ngb.predict(X_test)
            rmse_ngb = round(np.sqrt(mean_squared_error(y_test, y_pred_ngb)), 4)
            table.add_row(["NGBoost", "NA", est, rmse_ngb, time_ngb])
            pbar.update()

        # LightGBM
        for num_round in num_rounds:
            pbar.set_description(f"Building LightGBM w/ {num_round} Iterations")
            ltr = lgb.Dataset(X_train, y_train)
            start = default_timer()
            lgbm = lgb.train(
                lgbm_params,
                ltr,
                num_boost_round=num_round,
                valid_sets=[(ltr)],
                verbose_eval=0,
            )
            time_lgb = default_timer() - start
            y_pred_lgb = lgbm.predict(X_test)
            rmse_lgb = round(np.sqrt(mean_squared_error(y_test, y_pred_lgb)), 4)
            table.add_row(["LightGBM", f"{num_round}", "NA", rmse_lgb, time_lgb])
            pbar.update()

        # XGBoost
        for num_round in num_rounds:
            pbar.set_description(f"Building XGBoost w/ {num_round} Iterations")
            dtr, dte = (
                xgb.DMatrix(X_train, label=y_train),
                xgb.DMatrix(X_test, label=y_test),
            )
            start = default_timer()
            xgbst = xgb.train(xgb_params, dtr, num_round, verbose_eval=0)
            time_xgb = default_timer() - start
            y_pred_xgb = xgbst.predict(dte)
            rmse_xgb = round(np.sqrt(mean_squared_error(y_test, y_pred_xgb)), 4)
            table.add_row(["XGBoost", f"{num_round}", "NA", rmse_xgb, time_xgb])
            pbar.update()

        # Probability
        pbar.set_description("Plotting Probability Distribution")
        colors = ["olive", "navy", "tomato", "turquoise"]
        fig, ax = plt.subplots(1, 1)
        cands = np.sort(np.random.choice(np.arange(0, X_test.shape[0]), num_plots))
        for cand, c in zip(cands, colors):
            y_dists = ngb.pred_dist(X_test[cand,].reshape(1, -1))
            x_span = gen_xspan(y_dists.loc, y_dists.scale, num=100)
            dist_values = y_dists.pdf(x_span)
            ax.plot(x_span, dist_values, color=c, label=f"{cand}")
            ax.legend(loc="upper right")
            del y_dists, x_span, dist_values
        fig.canvas.draw()
        ax = normalize_ticks(ax)
        fig.suptitle("Probability Distribution")
        # Image will be saved in src/ in your host machine if you launched the container by docker-compse
        # Not valid for docker pull from dockerhub
        plt.savefig("prob_dist.png") if dockerenv else plt.show()
        plt.close()
        pbar.update()

        # Show Result
        pbar.set_description("All Done")
        pbar.write(str(table))
        pbar.update()


if __name__ == "__main__":
    benchmark()
