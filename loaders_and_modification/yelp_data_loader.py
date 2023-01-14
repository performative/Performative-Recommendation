import random
import pandas as pd
import torch
from torch.utils.data.dataset import T_co
from dataclasses import dataclass
from models import OneBitUserPreferenceEncoder
from loaders_and_modification.base_yelp_data_loader import BaseIterableYelpData


@dataclass
class BatchInfo:
    batch_user_ids: list
    batch_user_indexes: list
    user_pref_tensor: torch.tensor
    users_feature_tensor: torch.tensor


class IterableYelpData(BaseIterableYelpData):
    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, model: OneBitUserPreferenceEncoder, user_id_list: list[str], user_features_df: pd.DataFrame,
                 f_star, device: torch.cuda.device, batch_size: int, competition_frame):
        super(IterableYelpData).__init__()
        self.users_recommendation_df_dict = competition_frame.users_recommendation_df_dict
        self.restaurant_clients_dict = competition_frame.restaurant_clients_dict
        self.user_features_df = user_features_df
        self.user_id_list = user_id_list
        self.model = model
        self.batch_size = batch_size

        self.f_star = f_star.to(device)
        self.f_star.eval()
        self.f_star_dict_results = dict()
        self.device = device
        self.idx = 0
        self.batch_info = None
        self.user_hot_bit_df = model.get_df_ids(user_id_list)

    @staticmethod
    def create_lookup_table(source_list: list):
        return pd.DataFrame([torch.arange(0, len(source_list)).tolist()], columns=source_list)

    def start_batch(self):
        batch_user_ids = self.user_id_list[self.idx: self.idx + self.batch_size]
        lookup_user_index_table = IterableYelpData.create_lookup_table(batch_user_ids)
        users_feature_tensor = torch.from_numpy(self.user_features_df[batch_user_ids].to_numpy()).to(self.device).T
        users_hot_map_tensor = torch.from_numpy(self.user_hot_bit_df[batch_user_ids].to_numpy()).to(self.device).T
        user_pref_tensor = self.model.get_user_preference(users_hot_map_tensor)
        batch_user_indexes = lookup_user_index_table[batch_user_ids].values.tolist()[0]
        self.batch_info = BatchInfo(
            batch_user_ids,
            batch_user_indexes,
            user_pref_tensor,
            users_feature_tensor
        )
        self.idx += self.batch_size

    def build_sample(self):
        for user_id, user_index in zip(self.batch_info.batch_user_ids, self.batch_info.batch_user_indexes):
            items_features_of_user = torch.tensor(self.users_recommendation_df_dict[user_id].to_numpy(), device=self.device).T
            num_items = items_features_of_user.shape[0]
            user_pref = self.batch_info.user_pref_tensor[user_index].unsqueeze(0).repeat(num_items, 1)
            X = torch.cat((items_features_of_user, user_pref), dim=1)
            user_features = self.batch_info.users_feature_tensor[user_index]
            y = self.get_y_value_in_user_list(user_id, items_features_of_user, user_features, num_items)
            yield X, X, y


def split_user_train_test(user_abs_dict: dict, ratio: float = 0.7) -> (dict, dict):
    users_ids_list = list(user_abs_dict.keys())
    random.seed(10)
    random.shuffle(users_ids_list)
    separator = int(ratio*len(users_ids_list))
    user_ids_train, user_ids_test = set(users_ids_list[:separator]), set(users_ids_list[separator:])
    user_abs_train_dict, user_abs_test_dict = dict(), dict()
    for user_id in user_abs_dict.keys():
        if user_id in user_ids_train:
            user_abs_train_dict[user_id] = user_abs_dict[user_id]
        else:
            user_abs_test_dict[user_id] = user_abs_dict[user_id]
    return user_abs_train_dict, user_abs_test_dict
