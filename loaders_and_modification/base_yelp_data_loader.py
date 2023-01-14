from torch.utils.data import DataLoader, IterableDataset
from abc import abstractmethod
import functools
import operator
import pandas as pd
import torch


class BaseIterableYelpData(IterableDataset):
    @abstractmethod
    def start_batch(self):
        pass

    def check_if_finish(self):
        if self.idx >= len(self.user_id_list):
            self.idx = 0
            return True
        return False

    @abstractmethod
    def build_sample(self):
        pass

    def __iter__(self):
        return iter(self.build_sample())

    @staticmethod
    def create_lookup_table(source_list: list):
        return pd.DataFrame([torch.arange(0, len(source_list)).tolist()], columns=source_list)

    def get_relevant_users_and_items(self, batch_user_ids: list[str]) -> [list[str], list[str], list[list[str]]]:
        items_ordered_according_to_user = self.from_user_to_items_ids_df[
            batch_user_ids].T.values.tolist()  # should be list of list first element is a list of the items of the first user
        items_ids = list(set(functools.reduce(operator.iconcat, items_ordered_according_to_user, [])))
        all_relevant_users = set()
        for item_id in items_ids:
            all_relevant_users = all_relevant_users.union(set(self.restaurant_user_ids_dict[item_id]))
        all_relevant_users = list(all_relevant_users)
        return all_relevant_users, items_ids, items_ordered_according_to_user

    def get_y_value_in_user_list(self, user_id: str, items_features: torch.tensor, user_features: torch.tensor, num_items: int):
        '''

        :param user_id: The id of the user that we get the relevance of his items list
        :param items_features: The items in the recommendation list of the user. Tensor from the shape: (len list, num item features)
        :param user_features: The features of the user a 1D tensor that its length is the number of user features.
        :param num_items:
        :return:
        '''
        with torch.no_grad():
            if user_id not in self.f_star_dict_results:
                X_to_calc_y = torch.cat(
                    (items_features, user_features.unsqueeze(0).repeat(num_items, 1)), dim=1)
                y = self.f_star(X_to_calc_y.float()).squeeze(-1)
                self.f_star_dict_results[user_id] = y.detach().cpu()
            else:
                y = self.f_star_dict_results[user_id].to(self.device)
        return y
