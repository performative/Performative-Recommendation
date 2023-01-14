import sys
sys.path.append("..")
import random
import torch
import numpy as np
from collections import namedtuple
from loaders_and_modification.yelp_data_loader import IterableYelpData
from loaders_and_modification.strategic_data_loader import IterableStrategicYelpData
from models import OneBitUserPreferenceEncoder
from ndcg_div.ndcg_div_trainer import NdcgDivTrainer
import torch.optim as optim
from ndcg_div.ndcg_div_loss import Loss
import pandas as pd
import matplotlib.pyplot as plt
import os
from argparse import ArgumentParser
from synthetic_exp.user_overlap_exp import CompetitionFrameFromDict, get_connection_graph, change_graph
from synthetic_exp.synthetic_exp_utills import FStartSynthetic, create_random_df, RandomSyntheticCompetitionFrame, create_data_structures
from loaders_and_modification.post_items import get_post_items


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-a", dest='alpha', type=float, required=False, default=0)
    parser.add_argument("--list", dest='num_in_list', type=int, required=False, default=10)
    parser.add_argument("-nu", dest='num_users', type=int, required=False, default=200)
    parser.add_argument("-ni", dest='num_items', type=int, required=False, default=50)
    parser.add_argument("--lambda", dest='lambda_', type=float, required=False, default=0)
    parser.add_argument("--ustd", dest='u_std', type=float, required=False, default=0.1)
    parser.add_argument("--xstd", dest='x_std', type=float, required=False, default=1)
    parser.add_argument("--time", dest='time', type=int, required=False, default=10)
    parser.add_argument("--tau_sig", dest="tau_sig", type=float, required=False, default=2)
    parser.add_argument("--tau_div", dest="tau_div", type=float, required=False, default=5)
    parser.add_argument("--tau_ndcg", dest="tau_ndcg", type=float, required=False, default=5)
    parser.add_argument("--lr", dest="learn_rate", type=float, required=False, default=0.01)
    parser.add_argument("--dim", dest='dim', type=int, required=False, default=2)
    parser.add_argument("--k", dest='k', type=int, required=False, default=10)
    parser.add_argument("-ne", dest='num_epoch', type=int, required=False, default=500)
    parser.add_argument("--iter", dest='num_iters', type=int, required=False, default=1)
    parser.add_argument('--random_graph', dest='random_graph', action='store_true', required=False, default=False)
    parser.add_argument("--num_swap", dest='num_swap', type=int, required=False, default=0)
    return parser.parse_args()


class SyntheticDataMultipleRounds:
    def __init__(self, args):
        self.k = args.k if args.k is not None else args.num_in_list
        print(f'k: {self.k} \u03BB: {args.lambda_} x_std: {args.x_std} \u03B1={args.alpha}:')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.u_star = np.ones(args.dim) / np.sqrt(args.dim)
        self.batch_size = args.num_users
        self.f_star = FStartSynthetic(args.dim)
        self.loss = Loss(args.dim, top_k=self.k, sig_tau=args.tau_sig, tau_ndcg=args.tau_ndcg, tau_div=args.tau_div)
        self.all_list_loss = Loss(args.dim, top_k=args.num_in_list, sig_tau=args.tau_sig, tau_ndcg=args.tau_ndcg,
                             tau_div=args.tau_div)
        self.args = args

    def create_exp_data(self):
        if not self.args.random_graph:
            user_items_dict, items_id_list = get_connection_graph(self.args.num_users, self.args.num_items, self.args.num_in_list)
            if self.args.num_swap > 0:
                user_items_dict = change_graph(user_items_dict, self.args.num_swap, self.args.num_in_list)
            user_id_list = list(user_items_dict.keys())
            items_df = create_random_df(items_id_list, self.args.dim, self.args.x_std, -1 * self.u_star)
            pref_df = create_random_df(user_id_list, self.args.dim, self.args.u_std, self.u_star)
            competition_frame = CompetitionFrameFromDict(user_items_dict, items_df)
        else:
            items_id_list = [f'x{i}' for i in range(1, self.args.num_items + 1)]
            user_id_list = [f'u{i}' for i in range(1, self.args.num_users + 1)]
            pref_df, items_df, competition_frame = create_data_structures(items_id_list, user_id_list, self.args.dim,
                                                                          self.args.u_std,
                                                                          self.args.x_std, self.u_star, self.args.num_in_list)
        return pref_df, items_df, competition_frame, user_id_list

    def plot_distr(self, model: OneBitUserPreferenceEncoder, user_id_list: list[str], pref_df: pd.DataFrame,
                   items_df: pd.DataFrame, competition_frame, t: int):
        iter_data_strategic_test = IterableStrategicYelpData(model, user_id_list, pref_df, self.f_star, self.device,
                                                             self.batch_size,
                                                             items_df, competition_frame=competition_frame)
        iter_data_strategic_test.start_batch()
        post_items = iter_data_strategic_test.get_post_strategic_items_features(items_df.copy())
        alpha_plots_path = '../result/cost_plots'
        if t == 0:
            os.makedirs(alpha_plots_path, exist_ok=True)
            # items_df.to_csv(os.path.join(alpha_plots_path, 'items_t_0.csv'))
        # post_items.to_csv(os.path.join(alpha_plots_path, f'items_t_{2*t+1}.csv'))
        plt.figure(1, figsize=(8, 4))
        plt.title(f'\u03B1={self.args.alpha}: \u03BB: {self.args.lambda_} std: {self.args.u_std}')
        plt.subplot(121)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.title('before')
        plt.scatter(items_df.loc[0], items_df.loc[1], color='blue')
        plt.subplot(122)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.title('after')
        plt.scatter(post_items.loc[0], post_items.loc[1], color='red')
        plt.show()

    def get_move_res(self, model: OneBitUserPreferenceEncoder, user_id_list: list[str], pref_df: pd.DataFrame,
                     competition_frame, trainer: NdcgDivTrainer,
                     trainer_all_list_loss: Loss) -> (torch.tensor, torch.tensor):
        iter_data = IterableYelpData(model, user_id_list, pref_df, self.f_star, self.device, self.batch_size,
                                     competition_frame=competition_frame)
        before_move_res = trainer.predict(iter_data)
        before_move_all_list_res = trainer_all_list_loss.predict(iter_data)
        return before_move_res, before_move_all_list_res

    def run_2times(self, seed: int, items_df: pd.DataFrame, pref_df: pd.DataFrame,
                   competition_frame: RandomSyntheticCompetitionFrame, user_id_list: list[str], t: int):
        model = OneBitUserPreferenceEncoder(user_id_list, self.args.dim, alpha=self.args.alpha, seed=seed)
        optimizer = optim.Adam(model.parameters(), lr=self.args.learn_rate, betas=(0.9, 0.999))

        iter_data_train = IterableStrategicYelpData(model, user_id_list, pref_df, self.f_star, self.device,
                                                    self.batch_size, items_df, competition_frame=competition_frame)
        trainer = NdcgDivTrainer(model, self.loss, optimizer, self.device, self.f_star, lambda_=self.args.lambda_, batch_size=self.batch_size)
        trainer_all_list_loss = NdcgDivTrainer(model, self.all_list_loss, optimizer, self.device, self.f_star,
                                               lambda_=self.args.lambda_,
                                               batch_size=self.batch_size)
        ndcg_score_, div_score_, _ = trainer.fit(iter_data_train, self.args.num_epoch, print_every=1, get_best=True)
        before_move_res, before_move_all_list_res = self.get_move_res(model, user_id_list, pref_df, competition_frame, trainer, trainer_all_list_loss)
        iter_data = IterableStrategicYelpData(model, user_id_list, pref_df, self.f_star, self.device, self.batch_size, items_df,
                                              competition_frame=competition_frame)
        items_post_df = get_post_items(iter_data, items_df)
        competition_frame.update_items_df(items_post_df)
        after_move_res, after_move_res_all_list_res = self.get_move_res(model, user_id_list, pref_df, competition_frame, trainer, trainer_all_list_loss)

        two_times_res = namedtuple('two_timesr_res', ['ndcg_before_k', 'div_before_k', 'utility_before_k',
                                                      'ndcg_before_all_list', 'div_before_all_list',
                                                      'utility_before_all_list', 'ndcg_after_k', 'div_after_k',
                                                      'utility_after_k', 'ndcg_after_all_list', 'div_after_all_list',
                                                      'utility_after_all_list'])
        if self.args.dim == 2:
            self.plot_distr(model, user_id_list, pref_df, items_df, competition_frame, t)
        res = two_times_res(*(before_move_res[:3] + before_move_all_list_res[:3] + after_move_res[
                                                                                   :3] + after_move_res_all_list_res[
                                                                                         :3]))
        return res, items_post_df, competition_frame

    def run_single_iter(self, iteration: int):
        random.seed(iteration)
        np.random.seed(iteration)
        pref_df, items_df, competition_frame, user_id_list = self.create_exp_data()
        ndcg_k_list_per_time, div_k_list_per_time, utility_k_list_per_time = list(), list(), list()
        ndcg_all_list_per_time, div_all_list_per_time, utility_all_list_per_time = list(), list(), list()
        for t in range(self.args.time):
            result_2_times, items_df, competition_frame = self.run_2times(iteration, items_df, pref_df,
                                                                          competition_frame,
                                                                          user_id_list, t)
            ndcg_k_list_per_time += [result_2_times.ndcg_before_k, result_2_times.ndcg_after_k]
            ndcg_all_list_per_time += [result_2_times.ndcg_before_all_list, result_2_times.ndcg_after_all_list]
            div_k_list_per_time += [result_2_times.div_before_k, result_2_times.div_after_k]
            div_all_list_per_time += [result_2_times.div_before_all_list, result_2_times.div_after_all_list]
            utility_k_list_per_time += [result_2_times.utility_before_k, result_2_times.utility_after_k]
            utility_all_list_per_time += [result_2_times.utility_before_all_list, result_2_times.utility_after_all_list]

        print(f'k: {self.k} \u03BB: {self.args.lambda_} x_std: {self.args.x_std}')
        for t in range(2 * self.args.time):
            print(f'time: {t}', flush=True)
            print(f'ndcg_k: {ndcg_k_list_per_time[t]} div_k: {div_k_list_per_time[t]} utility_k: '
                  f'{utility_k_list_per_time[t]} ', flush=True)
            print(
                f'ndcg: {ndcg_all_list_per_time[t]} div: {div_all_list_per_time[t]} utility: {utility_all_list_per_time[t]} ',
                flush=True)

    def run_exp(self):
        for iteration in range(self.args.num_iters):
            self.run_single_iter(iteration)


if __name__ == '__main__':
    arguments = get_args()
    SyntheticDataMultipleRounds(arguments).run_exp()

