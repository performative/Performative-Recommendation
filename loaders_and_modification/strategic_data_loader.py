import pandas as pd
import torch
from torch.utils.data.dataset import T_co
from models import OneBitUserPreferenceEncoder
from dataclasses import dataclass
from loaders_and_modification.base_yelp_data_loader import BaseIterableYelpData


@dataclass
class BatchInfo:
    batch_user_ids: list
    items_ordered_according_to_user: list
    batch_user_indexes: list
    lookup_items_index_table: pd.DataFrame
    post_strategic_items_features: torch.tensor
    user_pref_tensor: torch.tensor
    users_feature_tensor: torch.tensor
    pre_strategic_items_features: torch.tensor


class IterableStrategicYelpData(BaseIterableYelpData):
    def __getitem__(self, index) -> T_co:
        pass

    def __init__(self, model: OneBitUserPreferenceEncoder, user_id_list: list[str], user_features_df: pd.DataFrame,
                 f_star, device: torch.cuda.device, batch_size: int, current_restaurant_df: pd.DataFrame,
                 competition_frame):
        super(IterableStrategicYelpData).__init__()
        self.users_recommendation_df_dict = competition_frame.users_recommendation_df_dict
        self.restaurant_user_ids_dict = competition_frame.restaurant_clients_dict
        self.from_user_to_items_ids_df = pd.DataFrame.from_dict(competition_frame.user_competition_business_ids)
        self.user_features_df = user_features_df
        self.user_id_list = user_id_list
        self.user_hot_bit_df = model.get_df_ids(user_id_list)
        self.idx = 0
        self.current_restaurant_df = current_restaurant_df
        self.items_df_before_move = current_restaurant_df
        self.f_star = f_star.to(device)
        self.f_star.eval()
        self.f_star_dict_results = dict()
        self.device = device
        self.batch_size = batch_size
        self.model = model
        self.batch_info = None


    @staticmethod
    def create_lookup_table(source_list: list) -> pd.DataFrame:
        return pd.DataFrame([torch.arange(0, len(source_list)).tolist()], columns=source_list)

    def start_batch(self):
        batch_user_ids = self.user_id_list[self.idx: self.idx+self.batch_size]
        all_relevant_users, items_ids, items_ordered_according_to_user = self.get_relevant_users_and_items(batch_user_ids)
        lookup_user_index_table = IterableStrategicYelpData.create_lookup_table(all_relevant_users)
        lookup_items_index_table = IterableStrategicYelpData.create_lookup_table(items_ids)
        users_feature_tensor = torch.from_numpy(self.user_features_df[all_relevant_users].to_numpy()).to(self.device).T
        users_hot_map_tensor = torch.from_numpy(self.user_hot_bit_df[all_relevant_users].to_numpy()).to(self.device).T
        user_pref_tensor = self.model.get_user_preference(users_hot_map_tensor)
        items_after_strategic_list = list()
        pre_strategic_items_features = torch.from_numpy(self.current_restaurant_df[items_ids].to_numpy()).to(self.device).T
        items_features_before_move = torch.from_numpy(self.items_df_before_move[items_ids].to_numpy()).to(self.device).T
        for index, item_id in enumerate(items_ids):
            users_rated_item_list = self.restaurant_user_ids_dict[item_id]
            users_rated_item_preference = user_pref_tensor[lookup_user_index_table[users_rated_item_list].values.tolist()]
            item_after_strategic = self.model.get_items_after_strategic(items_features_before_move[index], users_rated_item_preference)
            items_after_strategic_list.append(item_after_strategic)
        post_strategic_items_features = torch.stack(items_after_strategic_list)
        batch_user_indexes = lookup_user_index_table[batch_user_ids].values.tolist()[0]
        self.batch_info = BatchInfo(
                            batch_user_ids,
                            items_ordered_according_to_user,
                            batch_user_indexes,
                            lookup_items_index_table,
                            post_strategic_items_features,
                            user_pref_tensor,
                            users_feature_tensor,
                            pre_strategic_items_features
        )
        self.idx += self.batch_size

    def get_post_strategic_items_features(self, items_df_before_move: pd.DataFrame):
        items_that_moved = list(self.batch_info.lookup_items_index_table.columns)
        lookup_indexes_list = self.batch_info.lookup_items_index_table.values.tolist()[0]
        post_strategic_data = self.batch_info.post_strategic_items_features[lookup_indexes_list].cpu().detach().numpy()
        items_df_before_move[items_that_moved] = post_strategic_data.T
        return items_df_before_move

    def build_sample(self):
        for user_id, user_items_ids, user_index in zip(self.batch_info.batch_user_ids, self.batch_info.items_ordered_according_to_user, self.batch_info.batch_user_indexes):
            items_of_user_indexes = self.batch_info.lookup_items_index_table[user_items_ids].values.tolist()[0]
            items_features_of_user_post_strategic = self.batch_info.post_strategic_items_features[items_of_user_indexes]
            user_pref = self.batch_info.user_pref_tensor[user_index]
            num_items = items_features_of_user_post_strategic.shape[0]
            X_post_strategic = torch.cat((items_features_of_user_post_strategic, user_pref.unsqueeze(0).repeat(num_items, 1)), dim=1)
            items_features_of_user_pre_strategic = self.batch_info.pre_strategic_items_features[items_of_user_indexes]
            user_features = self.batch_info.users_feature_tensor[user_index]
            X_pre_strategic = torch.cat((items_features_of_user_pre_strategic, user_pref.unsqueeze(0).repeat(num_items, 1)), dim=1)
            y = self.get_y_value_in_user_list(user_id, items_features_of_user_pre_strategic, user_features, num_items)
            yield X_pre_strategic, X_post_strategic, y




