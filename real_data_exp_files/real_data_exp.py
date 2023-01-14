import sys
sys.path.append("..")
import torch.optim as optim
from ndcg_div.ndcg_div_trainer import TargetNdcgDivTrainer, TargetTrainerEarlyStopParam
from models import OneBitUserPreferenceEncoder, FStarWarraper
from ndcg_div.ndcg_div_loss import Loss
from real_data_exp_files.f_star_code.f_star_creator import get_f_star
from real_data_exp_files.parse_params import parse_args
import os
from loaders_and_modification.post_items import get_post_items
from real_data_exp_files.utills import *
from real_data_exp_files.data_creation.competition_frame import *
from copy import deepcopy
from math import ceil, floor


class RealDataExp:
    def __init__(self, args):
        self.alpha = args.alpha
        self.target_ndcg = args.ndcg_target
        self.k = args.k

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.user_features_df = pd.read_csv(user_features_df_path).drop(columns=['Unnamed: 0'])
        self.users_ids_list = list(self.user_features_df.columns)
        self.orig_df = pd.read_csv(restaurant_df_path)
        self.current_rest_df = self.orig_df.copy()

        self.competition_frame_full = load_competition_frame()
        self.competition_frame_train = deepcopy(self.competition_frame_full)
        self.competition_frame_val = deepcopy(self.competition_frame_full)
        self.competition_frame_test = deepcopy(self.competition_frame_full)

        self.num_items_features = self.current_rest_df.shape[0]
        self.f_star = FStarWarraper(get_f_star(self.orig_df, force_to_train=False))
        self.loss = Loss(self.num_items_features, top_k=args.k, sig_tau=args.tau_sig, tau_ndcg=args.tau_ndcg,
                    tau_div=args.tau_div)
        self.strategic_model = args.strategic_model
        self.saving_model_folder = RealDataExp.create_saving_path(models_yelp, self.k, self.alpha, self.strategic_model, self.target_ndcg)
        self.saving_strategic_data_folder = RealDataExp.create_saving_path(strategic_data_folder, self.k, self.alpha, self.strategic_model, self.target_ndcg)
        self.args = args

        self.target_trainer_early_stop_p = TargetTrainerEarlyStopParam(
            min_epoch_to_train=50,
            target_current_max_diff=0.035,
            current_target_max_diff=0.01,
            last_taring_period=160,
            last_taring_period_target_current_max_diff=0.025
        )
        self.line_search_params = LambdaLineSearch(
            lambda_high=120,
            lambda_low=0.0,
            num_tries=8,
            tol_div_imp=0.005,
            ndcg_target_thresh=0.01,
            lambda_low_lim_to_stop=0.001,
            lambda_high_lim_to_stop=100.001
        )

    # todo: might be in utills
    @staticmethod
    def get_base_folder_according_exp_params(k, alpha):
        return f'k_{k}_{from_float_to_str(alpha)}'

    @staticmethod
    def create_saving_path(base_folder, k, alpha, strategic_model, target_ndcg=None, random=False):
        strategic_model_str = 'strategic_model' if strategic_model else 'non_strategic_model'
        if random:
            strategic_model_str += 'random'
        exp_params_folder = RealDataExp.get_base_folder_according_exp_params(k, alpha)
        if target_ndcg:
            saving_folder = os.path.join(base_folder, exp_params_folder, from_float_to_str(target_ndcg),
                                         strategic_model_str)
        else:
            saving_folder = os.path.join(base_folder, exp_params_folder, strategic_model_str)
        os.makedirs(saving_folder, exist_ok=True)
        return saving_folder

    def train_current_lambda(self, lambda_, thresh, target_ndcg=None):
        print(f'trying \u03BB: {lambda_}')
        model = OneBitUserPreferenceEncoder(self.users_ids_list, self.num_items_features, self.alpha).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.args.learn_rate, betas=(0.9, 0.999))
        trainer = TargetNdcgDivTrainer(model, self.loss, optimizer, self.device, self.f_star, lambda_, self.args.batch_size,
                                 self.args.strategic_model)

        iter_data_train = get_data_loader(model, self.users_ids_list, self.user_features_df, self.f_star, self.device,
                                          self.current_rest_df, self.args, self.args.batch_size, self.competition_frame_train,
                                          train_mode=True)
        iter_data_val = get_data_loader(model, self.users_ids_list, self.user_features_df, self.f_star, self.device,
                                          self.current_rest_df, self.args, self.args.batch_size,
                                          self.competition_frame_val,
                                          train_mode=True)
        ndcg_score, div_score, best_model_state = trainer.target_ndcg_fit(iter_data_train, iter_data_val,
                                                                          self.args.num_epoch, thresh,
                                                                          self.target_trainer_early_stop_p,
                                                                          print_every=1, target_ndcg=target_ndcg)
        return ndcg_score, div_score, best_model_state


    def check_ndcg_target_err(self, ndcg):
        return abs(ndcg - self.target_ndcg) <= self.line_search_params.ndcg_target_thresh

    def look_model_for_target_ndcg(self, current_lambda):
        print(f'looking for ndcg close to {self.target_ndcg}')
        lambda_low, lambda_high = self.line_search_params.lambda_low, self.line_search_params.lambda_high
        most_fit_ndcg = None
        most_fit_div = -1
        most_fit_model_state_dict = None
        most_fit_lambda = None
        for i in range(self.line_search_params.num_tries):
            '''
            The use in this method leads us to find the model that his best loss while training has ndcg that is the
             closest to target ndcg. We need to find model that his ndcg is in range from target and has maximum diversity!
            '''
            print(f'current lambda: {current_lambda}')
            most_fit_prev_div = most_fit_div
            ndcg_score, div_score, current_model_state = self.train_current_lambda(current_lambda, self.line_search_params.ndcg_target_thresh, self.target_ndcg)
            print(f'tried \u03BB {current_lambda} and got ndcg: {ndcg_score} and div: {div_score}')
            in_thresh = self.check_ndcg_target_err(ndcg_score)
            if most_fit_ndcg is None or in_thresh:
                if most_fit_div < div_score:
                    most_fit_ndcg, most_fit_div, most_fit_model_state_dict = ndcg_score, div_score, current_model_state
                    most_fit_lambda = current_lambda
                    if div_score - most_fit_prev_div < self.line_search_params.tol_div_imp:
                        break

            elif abs(ndcg_score - self.target_ndcg) < abs(most_fit_ndcg - self.target_ndcg):
                most_fit_ndcg, most_fit_div, most_fit_model_state_dict = ndcg_score, div_score, current_model_state
                most_fit_lambda = current_lambda

            if in_thresh or ndcg_score > self.target_ndcg:
                lambda_low = current_lambda
                current_lambda = (current_lambda + lambda_high) / 2
            else:
                lambda_high = current_lambda
                current_lambda = (current_lambda + lambda_low) / 2

            if current_lambda < self.line_search_params.lambda_low_lim_to_stop or current_lambda > self.line_search_params.lambda_high_lim_to_stop:
                break

        print(f'find ndcg:{most_fit_ndcg} with div: {most_fit_div} and \u03BB: {most_fit_lambda}')
        model = OneBitUserPreferenceEncoder(self.users_ids_list, self.num_items_features, self.alpha).to(
            self.device)
        model.load_state_dict(most_fit_model_state_dict)
        return model, most_fit_lambda

    @staticmethod
    def get_model_name(saving_model_folder: str, time: int):
        return os.path.join(saving_model_folder, f'model_time_{time}.pt')

    @staticmethod
    def get_strategic_data_name(saving_strategic_data_folder: str, time: int):
        return os.path.join(saving_strategic_data_folder, f'data_time_{time}.csv')

    def run_single_time_strategic_move(self, target_ndcg_model, time):
        iter_data = IterableStrategicYelpData(target_ndcg_model, self.users_ids_list, self.user_features_df,
                                              self.f_star,
                                              self.device, len(self.users_ids_list), self.current_rest_df,
                                              self.competition_frame_full)

        strategic_data_saving_path = RealDataExp.get_strategic_data_name(self.saving_strategic_data_folder, time+1)
        self.current_rest_df = get_post_items(iter_data, self.current_rest_df, strategic_data_saving_path)

    def run_single_time(self, time: int, lambda_to_start: float):
        target_ndcg_model, most_fit_lambda = self.look_model_for_target_ndcg(lambda_to_start)
        torch.save(target_ndcg_model.state_dict(), RealDataExp.get_model_name(self.saving_model_folder, time))
        self.run_single_time_strategic_move(target_ndcg_model, time)
        return most_fit_lambda

    def dump_lambda_used(self, lambda_found_list: list):
        print('lambda that used:')
        print(lambda_found_list)
        d = {'lambda_list': lambda_found_list}
        with open(os.path.join(self.saving_model_folder, 'lambda_used.json'), 'w+') as f:
            json.dump(d, f, indent=4)

    def run_range_times(self, lambda_to_start=5.0):
        if self.target_ndcg == 2.0:
            lambda_to_start = 0.0
        elif self.target_ndcg > 1.0:
            lambda_to_start = 0.1
        lambda_found_list = list()
        if self.args.start_time != 0:
            strategic_data_saving_path = RealDataExp.get_strategic_data_name(self.saving_strategic_data_folder,
                                                                             self.args.start_time)
            self.current_rest_df = pd.read_csv(strategic_data_saving_path)

        for time in range(self.args.start_time, self.args.end_time):
            print(f'time: {time}!!!!!!!!!!!!!!!!')
            self.competition_frame_full.adjust_competition_frame_according_len_rec(self.current_rest_df,
                                                                                   len_rec_list_rest,
                                                                                   seed=time)
            self.competition_frame_train.adjust_competition_frame_according_len_rec(self.current_rest_df,
                                                                                    len_rec_list_train,
                                                                                    seed=time)
            self.competition_frame_val.adjust_competition_frame_according_len_rec(self.current_rest_df,
                                                                                  len_rec_list_val,
                                                                                  seed=time)
            lambda_found = self.run_single_time(time, lambda_to_start)
            lambda_found_list.append(lambda_found)
        self.dump_lambda_used(lambda_found_list)

    def calc_strategic_move_for_all_time(self):
        model = OneBitUserPreferenceEncoder(self.users_ids_list, self.num_items_features, self.alpha).to(self.device)
        for time in range(self.args.start_time, self.args.end_time):
            print(f'current time: {time}')
            model.load_state_dict(torch.load(RealDataExp.get_model_name(self.saving_model_folder, time)))
            self.run_single_time_strategic_move(model, time)

    def calc_random_ndcg_div(self, model, data):
        ndcg_list, div_list, utility_list = [], [], []
        for _ in range(10):
            trainer = TargetNdcgDivTrainer(model, self.loss, None, self.device, self.f_star, 0, self.args.batch_size,
                                           self.args.strategic_model)
            data_iter = IterableYelpData(model, self.users_ids_list, self.user_features_df, self.f_star, self.device,
                                         self.args.batch_size, self.competition_frame_test)
            ndcg_score, diversity_score, utility_score = trainer.get_random_ndcg_div(data_iter)
            ndcg_list.append(ndcg_score)
            div_list.append(diversity_score)
            utility_list.append(utility_score)


        return torch.tensor(ndcg_list).mean().item(), torch.tensor(div_list).mean().item(), \
               torch.tensor(utility_list).mean().item()

    def load_data_in_time(self, time: int, target_ndcg: float):
        if time != 0:
            data_folder = RealDataExp.create_saving_path(strategic_data_folder, self.k, self.alpha,
                                                         self.strategic_model, target_ndcg)
            saving_data_path = RealDataExp.get_strategic_data_name(data_folder, ceil((time) / 2))
            print(saving_data_path)
            data = pd.read_csv(saving_data_path)
        else:
            data = self.orig_df
        return data

    @staticmethod
    def dump_result_dict(result_ndcg_list: list, result_div_list: list, result_utility_list: list, output_path: str):
        result_dict = dict()
        result_dict['ndcg_not_sorted'] = result_ndcg_list
        result_dict['div_not_sorted'] = result_div_list
        result_dict['utility_not_sorted'] = result_utility_list
        result_dict['div_sorted_according_div'] = sorted(result_div_list)
        result_dict['ndcg_sorted_according_div'] = [ndcg for ndcg, div in sorted(zip(result_ndcg_list, result_div_list),
                                                                                 key=lambda tup: tup[1])]
        result_dict['utility_sorted_according_div'] = [ndcg for ndcg, div in
                                                       sorted(zip(result_utility_list, result_div_list),
                                                              key=lambda tup: tup[1])]
        with open(output_path, 'w+') as f:
            json.dump(result_dict, f, indent=4)

    def run_single_time_test(self, time: int, ndcg_target_list: list, output_path: str, random: bool = False):
        result_ndcg_list, result_div_list, result_utility_list = list(), list(), list()
        model = OneBitUserPreferenceEncoder(self.users_ids_list, self.num_items_features, self.alpha).to(self.device)
        print(f'time: {time}')
        for target_ndcg in ndcg_target_list:
            model_folder = RealDataExp.create_saving_path(models_yelp, self.k, self.alpha, self.strategic_model, target_ndcg)
            try:
                model.load_state_dict(torch.load(RealDataExp.get_model_name(model_folder, floor(time/2))))
            except:
                print(f'model is not exist for {target_ndcg} and time {time}')
                continue
            data = self.load_data_in_time(time, target_ndcg)
            self.competition_frame_test.adjust_competition_frame_according_len_rec(data,
                                                                                   len_rec_list_test,
                                                                                   seed=floor((time) / 2))
            if random:
                ndcg_score, diversity_score, utility_score = self.calc_random_ndcg_div(model, data)
            else:
                optimizer = optim.Adam(model.parameters(), lr=self.args.learn_rate, betas=(0.9, 0.999))
                trainer = TargetNdcgDivTrainer(model, self.loss, optimizer, self.device, self.f_star, 0, self.args.batch_size, self.args.strategic_model)
                data_iter = IterableYelpData(model, self.users_ids_list, self.user_features_df, self.f_star, self.device,
                                              self.args.batch_size, self.competition_frame_test)

                ndcg_score, diversity_score, utility_score, loss = trainer.predict(data_iter)
            print(f'target_ndcg: {target_ndcg}: ndcg: {ndcg_score} diversity: {diversity_score}')
            result_ndcg_list.append(ndcg_score)
            result_div_list.append(diversity_score)
            result_utility_list.append(utility_score)
        RealDataExp.dump_result_dict(result_ndcg_list, result_div_list, result_utility_list, output_path)

    def test_model_for_all_time(self, ndcg_target_list, random=False):
        out_folder = RealDataExp.create_saving_path(result_folder, self.k, self.alpha, self.strategic_model, random=random)
        for time in range(self.args.start_time, 2*self.args.end_time):
            output_path = os.path.join(out_folder, f'result_time_{time}.json')
            self.run_single_time_test(time, ndcg_target_list, output_path, random=random)


if __name__ == '__main__':
    parsed_args = parse_args()
    if parsed_args.calc_post_items:
        RealDataExp(parsed_args).calc_strategic_move_for_all_time()

    elif parsed_args.test:
        RealDataExp(parsed_args).test_model_for_all_time(parsed_args.ndcg_target_list, parsed_args.random)
    else:
        RealDataExp(parsed_args).run_range_times()
