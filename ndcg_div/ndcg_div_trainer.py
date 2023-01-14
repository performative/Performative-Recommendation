import torch
from collections import namedtuple
from torch.utils.data import DataLoader
from collections import defaultdict, OrderedDict
import json
from copy import deepcopy
from dataclasses import dataclass
from ndcg_div.ndcg_div_loss import Loss
from loaders_and_modification.base_yelp_data_loader import BaseIterableYelpData


class NdcgDivTrainer:
    def __init__(self, model: torch.nn.Module, loss: Loss, optimizer: torch.optim.Optimizer, device: torch.cuda.device,
                 f_star: torch.nn.Module, lambda_: int, batch_size: int, num_with_no_imp: int = 2000):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.model = self.model.to(self.device)
        self.f_star = f_star
        self.f_star.eval()
        self.lambda_ = lambda_
        self.batch_size = batch_size
        self.max_num_with_no_imp = num_with_no_imp
        self.epoch_res = namedtuple('EpochRes', ['ndcg_loss', 'ndcg_score', 'diversity_loss', 'diversity_score',
                                                 'utility_score', 'loss'])
        self.result = defaultdict(list)

    def for_each_batch(self, X: torch.tensor, x_for_calc_y: torch.tensor, y: torch.tensor, ndcg_score_list: list[float],
                       approx_ndcg_score_list: list[float], approx_diversity_list: list[float], diversity_score_list: list[float],
                       loss_list: list[float], utility_list: list[float], is_training: bool):
        model_score_for_div = self.model(X.float()).squeeze(2)
        model_score_for_ndcg = self.model(x_for_calc_y.float()).squeeze(2)
        approx_ndcg_score, ndcg_score, approx_diversity, diversity_score = self.loss(X, model_score_for_div, model_score_for_ndcg, y)
        # print(approx_ndcg_score.mean().item(), ndcg_score.mean().item(), approx_diversity.mean().item(), diversity_score.mean().item())
        utility_list.extend(y.mean(dim=1).tolist())
        approx_diversity_list.extend(approx_diversity.tolist())
        diversity_score_list.extend(diversity_score.tolist())
        ndcg_score_list.extend(ndcg_score.cpu().tolist())
        approx_ndcg_score_list.extend(approx_ndcg_score.cpu().tolist())

        loss = -1*(approx_ndcg_score + self.lambda_ * approx_diversity).mean()

        if is_training:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        loss_list.append(loss.item())

    def run_all_batch(self, data_loader:DataLoader, ndcg_score_list: list[float], approx_ndcg_score_list: list[float],
                      approx_diversity_list: list[float], diversity_score_list: list[float], loss_list: list[float],
                      utility_list: list[float], is_training: bool):
        while not data_loader.dataset.check_if_finish():
            data_loader.dataset.start_batch()
            X_for_ndcg, X_for_div, y = next(iter(data_loader))
            self.for_each_batch(X_for_div, X_for_ndcg, y, ndcg_score_list, approx_ndcg_score_list, approx_diversity_list,
                                diversity_score_list, loss_list, utility_list, is_training)

    def for_each_epoch(self, data_loader: DataLoader, is_training: bool = False) -> namedtuple:
        ndcg_score_list = list()
        approx_ndcg_score_list = list()
        approx_diversity_list = list()
        diversity_score_list = list()
        utility_list = list()
        loss_list = list()
        if is_training:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()
        self.run_all_batch(data_loader, ndcg_score_list, approx_ndcg_score_list, approx_diversity_list,
                                   diversity_score_list, loss_list, utility_list, is_training)
        return self.epoch_res(sum(approx_ndcg_score_list)/len(approx_ndcg_score_list),
                              sum(ndcg_score_list)/ len(ndcg_score_list),
                              sum(approx_diversity_list) / len(approx_diversity_list),
                              sum(diversity_score_list) / len(diversity_score_list),
                              sum(utility_list) / len(utility_list),
                              sum(loss_list)/len(loss_list))

    def print_epoch_res(self, epoch: int, train_epoch_res: namedtuple, test_epoch_res: namedtuple):
        print(
            f'epoch: {epoch} train loss: {train_epoch_res.loss} train loss ndcg: {train_epoch_res.ndcg_loss} train ndcg score: {train_epoch_res.ndcg_score}\n'
            f'train loss diversity: {train_epoch_res.diversity_loss} train diversity score: {train_epoch_res.diversity_score}', flush=True)

    def fit(self, iter_data_train: BaseIterableYelpData, num_epoch: int, print_every: int = 1, get_best: bool = False)\
            -> (float, float, OrderedDict):
        data_loader_train = DataLoader(iter_data_train, batch_size=self.batch_size, num_workers=0)
        num_with_no_imp = 0
        current_best_loss = float('inf')

        best_result, best_loss = None, None
        for epoch in range(num_epoch):
            train_epoch_res = self.for_each_epoch(data_loader_train, is_training=True)
            if epoch % print_every == 0 or epoch == num_epoch - 1:
                self.print_epoch_res(epoch, train_epoch_res, None)
            if train_epoch_res.loss < current_best_loss:
                num_with_no_imp = 0
                current_best_loss = train_epoch_res.loss
                best_result = (train_epoch_res.ndcg_score, train_epoch_res.diversity_score, self.model.state_dict())
            else:
                num_with_no_imp += 1
                if num_with_no_imp == self.max_num_with_no_imp:
                    break
        if get_best:
            return best_result
        return train_epoch_res.ndcg_score, train_epoch_res.diversity_score, self.model.state_dict()

    def dump_result(self, result_path: str):
        with open(result_path, 'w+') as f:
            json.dump(self.result, f, indent=4)

    def predict(self, iter_data_test: BaseIterableYelpData) -> (float, float, float, float):
        data_loader_test = DataLoader(iter_data_test, batch_size=self.batch_size, num_workers=0)
        test_epoch_res = self.for_each_epoch(data_loader_test, is_training=False)
        loss = -1 * (test_epoch_res.ndcg_loss + self.lambda_ * test_epoch_res.diversity_loss) if self.lambda_ is not None else None
        return test_epoch_res.ndcg_score, test_epoch_res.diversity_score, test_epoch_res.utility_score, loss

@dataclass
class TargetTrainerEarlyStopParam:
    min_epoch_to_train: int
    target_current_max_diff: float
    current_target_max_diff: float
    # from which epoch the stopping criterion is more stringent
    last_taring_period: float
    last_taring_period_target_current_max_diff: float


class TargetNdcgDivTrainer(NdcgDivTrainer):
    def print_epoch_res(self, epoch: int, train_epoch_res: namedtuple, val_epoch_res: namedtuple):
        print(f'epoch: {epoch}')
        print(
            f'train loss: {train_epoch_res.loss} train loss ndcg: {train_epoch_res.ndcg_loss} train ndcg score: {train_epoch_res.ndcg_score}\n'
            f'train loss diversity: {train_epoch_res.diversity_loss} train diversity score: {train_epoch_res.diversity_score}', flush=True)
        print(
            f'val loss: {val_epoch_res.loss} val loss ndcg: {val_epoch_res.ndcg_loss} val ndcg score: {val_epoch_res.ndcg_score}\n'
            f'val loss diversity: {val_epoch_res.diversity_loss} val diversity score: {val_epoch_res.diversity_score}',
            flush=True)

    def target_ndcg_fit(self, iter_data_train: BaseIterableYelpData, iter_data_val: BaseIterableYelpData, num_epoch: int,
                        thresh: float, early_stop_params: TargetTrainerEarlyStopParam, print_every: int = 1,
                        target_ndcg=None):
        data_loader_train = DataLoader(iter_data_train, batch_size=self.batch_size, num_workers=0)
        data_loader_val = DataLoader(iter_data_val, batch_size=self.batch_size, num_workers=0)
        current_best_div_in_range, best_result, most_fit_ndcg = None, None, float('inf')
        for epoch in range(num_epoch):
            train_epoch_res = self.for_each_epoch(data_loader_train, is_training=True)
            val_epoch_res = self.for_each_epoch(data_loader_val, is_training=False)
            if current_best_div_in_range is None:
                if abs(most_fit_ndcg - target_ndcg) > abs(val_epoch_res.ndcg_score - target_ndcg):
                    most_fit_ndcg = val_epoch_res.ndcg_score
                    best_result = (val_epoch_res.ndcg_score, val_epoch_res.diversity_score, deepcopy(self.model.state_dict()))
            if epoch >= early_stop_params.min_epoch_to_train and target_ndcg:
                if target_ndcg - best_result[0] > early_stop_params.target_current_max_diff or \
                        best_result[0] - target_ndcg > early_stop_params.current_target_max_diff or \
                        (epoch >= early_stop_params.last_taring_period and target_ndcg - best_result[
                            0] > early_stop_params.last_taring_period_target_current_max_diff):
                    print('Early termination of the training due to incompatibility of the results with the desired ndcg')
                    break

            if epoch % print_every == 0 or epoch == num_epoch - 1:
                self.print_epoch_res(epoch, train_epoch_res, val_epoch_res)
            if abs(target_ndcg - val_epoch_res.ndcg_score) < thresh:
                if current_best_div_in_range is None or current_best_div_in_range < val_epoch_res.diversity_score:
                    current_best_div_in_range = val_epoch_res.diversity_score
                    best_result = (val_epoch_res.ndcg_score, val_epoch_res.diversity_score, deepcopy(self.model.state_dict()))

        return best_result

    def get_random_ndcg_div(self, iter_data_test: BaseIterableYelpData):
        data_loader_test = DataLoader(iter_data_test, batch_size=self.batch_size, num_workers=0)
        ndcg_score_list = list()
        diversity_score_list = list()
        utility_list = list()
        while not data_loader_test.dataset.check_if_finish():
            data_loader_test.dataset.start_batch()
            X_for_ndcg, X_for_div, y = next(iter(data_loader_test))
            random_score = torch.rand_like(y)
            _, ndcg_score, _, diversity_score = self.loss(X_for_ndcg, random_score, random_score, y)
            ndcg_score_list.extend(ndcg_score.detach().cpu().tolist())
            utility_list.extend(y.mean(dim=1).tolist())
            diversity_score_list.extend(diversity_score.detach().cpu().tolist())
        ndcg_score = torch.tensor(ndcg_score_list).mean().cpu().item()
        diversity_score = torch.tensor(diversity_score_list).mean().cpu().item()
        utility_score = torch.tensor(utility_list).mean().cpu().item()
        return ndcg_score, diversity_score, utility_score

