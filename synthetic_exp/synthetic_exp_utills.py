import torch
import numpy as np
import pandas as pd
import random
from collections import defaultdict


class FStartSynthetic(torch.nn.Module):
    def __init__(self, num_items_features: int):
        super().__init__()
        self.num_items_features = num_items_features

    def split_x_to_user_items_features(self, x: torch.tensor) -> (torch.tensor, torch.tensor):
        items_features, user_features = x[:, :self.num_items_features], x[0, self.num_items_features:]
        return items_features, user_features

    def __call__(self, x: torch.tensor) -> torch.tensor:
        items_features, user_features = self.split_x_to_user_items_features(x)
        scores = items_features @ user_features.unsqueeze(1) / user_features.norm()
        return 2**scores


class CompetitionFrameFromDict:
    def __init__(self, user_items_dict: dict, items_df: pd.DataFrame):
        '''

        :param user_items_dict: dictionary that the key is user and the value is list of item_id
        '''
        self.user_competition_business_ids = user_items_dict
        self.restaurant_clients_dict = defaultdict(list)
        self.users_recommendation_df_dict = dict()
        user_id_list = list(user_items_dict.keys())
        num_items = len(user_items_dict[user_id_list[0]])
        for u_id in user_id_list:
            assert len(user_items_dict[u_id]) == num_items, 'not all users have the same number of items!'
            self.users_recommendation_df_dict[u_id] = items_df[self.user_competition_business_ids[u_id]]
            for x_id in user_items_dict[u_id]:
                self.restaurant_clients_dict[x_id].append(u_id)

    def update_items_df(self, items_post_df: pd.DataFrame):
        for u_key in self.user_competition_business_ids.keys():
            self.users_recommendation_df_dict[u_key] = items_post_df[self.user_competition_business_ids[u_key]]


class RandomSyntheticCompetitionFrame:
    def __init__(self, items_df: pd.DataFrame, num_in_list: int, user_id_list: list):
        self.user_competition_business_ids = dict()
        id_list = list(items_df.columns)
        assert len(id_list) >= num_in_list, 'number of items in list is higher than the number of items'
        for u in user_id_list:
            random.shuffle(id_list)
            self.user_competition_business_ids[u] = id_list[:num_in_list]

        self.users_recommendation_df_dict = dict()
        for u_key in self.user_competition_business_ids.keys():
            self.users_recommendation_df_dict[u_key] = items_df[self.user_competition_business_ids[u_key]]

        self.restaurant_clients_dict = defaultdict(list)
        for user_id in self.user_competition_business_ids.keys():
            for business_id in self.user_competition_business_ids[user_id]:
                self.restaurant_clients_dict[business_id].append(user_id)

        self.num_clients_dict = defaultdict(list)
        for rest_id, clients_list in self.restaurant_clients_dict.items():
            self.num_clients_dict[len(clients_list)].append(rest_id)

    def update_items_df(self, items_post_df: pd.DataFrame):
        for u_key in self.user_competition_business_ids.keys():
            self.users_recommendation_df_dict[u_key] = items_post_df[self.user_competition_business_ids[u_key]]


def create_random_df(id_list: list[str], num_dim: int, std: float, p_star: np.array) -> pd.DataFrame:
    data = np.random.normal(p_star, std, (len(id_list), num_dim)).T
    data = data / np.linalg.norm(data, axis=0)
    df = pd.DataFrame(data=data, columns=id_list)
    return df


def create_data_structures(items_id_list: list[str], user_id_list: list[str], dim: int, u_std: float, x_std: float,
                           p_star: np.array, num_in_list: int) -> (pd.DataFrame, pd.DataFrame, RandomSyntheticCompetitionFrame):
    pref_df = create_random_df(user_id_list, dim, u_std, p_star)
    items_df = create_random_df(items_id_list, dim, x_std, p_star)
    competition_frame = RandomSyntheticCompetitionFrame(items_df, num_in_list, user_id_list)
    return pref_df, items_df, competition_frame
