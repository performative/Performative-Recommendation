from real_data_exp_files.definitions import *
import pandas as pd
from real_data_exp_files.data_creation.user_avg_content import filter_user_few_rev
from collections import defaultdict
import json
import dill
import heapq
import random
from copy import deepcopy


class ItemQNode:
    def __init__(self):
        self.user_set = set()
        self.item_id = None

    def add(self, user_id):
        self.user_set.add(user_id)

    def remove(self, user_id):
        self.user_set.remove(user_id)

    def set_item_id(self, item_id):
        self.item_id = item_id

    def __lt__(self, other):
        return len(self.user_set) > len(other.user_set)


class CompetitionFrame:
    @staticmethod
    def create_items_heap(user_business_dict: dict) -> (list, dict):
        heap_dict = defaultdict(ItemQNode)
        for u_id in user_business_dict.keys():
            for item_id in user_business_dict[u_id]:
                heap_dict[item_id].add(u_id)
        items_heap = list()
        for item_id, item_node in heap_dict.items():
            item_node.set_item_id(item_id)
            heapq.heappush(items_heap, item_node)
        return items_heap, heap_dict

    @staticmethod
    def calc_users_competition_restaurant(user_business_dict: dict, num_items_in_list: int) -> dict:
        items_heap, heap_dict = CompetitionFrame.create_items_heap(user_business_dict)
        user_competition_business = defaultdict(list)
        user_not_full = set(user_business_dict.keys())
        while len(user_not_full):
            item_node = heapq.heappop(items_heap)
            item_user_list = list(item_node.user_set)
            should_sort = False
            for u_id in item_user_list:
                user_competition_business[u_id].append(item_node.item_id)
                if len(user_competition_business[u_id]) == num_items_in_list:
                    user_not_full.remove(u_id)
                    should_sort = True
                    for item in user_business_dict[u_id]:
                        heap_dict[item].remove(u_id)
            if should_sort:
                items_heap.sort(reverse=False) # it should be false since we switch > operator
        return user_competition_business

    def create_for_each_user_competition_data(self, restaurant_df: pd.DataFrame, num_items_in_list: int, seed: int) -> (dict, dict):
        user_business_competition_data_frames_dict = dict()
        user_competition_business_ids = dict()
        for user_id, business_id_list in self.full_user_competition_business_ids.items():
            copy_business_id_list = deepcopy(business_id_list)
            random.seed(seed)
            random.shuffle(copy_business_id_list)
            user_competition_business_ids[user_id] = copy_business_id_list[:num_items_in_list]
            user_business_competition_data_frames_dict[user_id] = restaurant_df[copy_business_id_list[:num_items_in_list]]
        return user_business_competition_data_frames_dict, user_competition_business_ids

    def get_restaurant_to_competitions_mapping(self) -> dict:
        restaurant_user_id_dict = defaultdict(list)
        for user_id in self.users_recommendation_df_dict.keys():
            for business_id in self.users_recommendation_df_dict[user_id].columns:
                restaurant_user_id_dict[business_id].append(user_id)
        return restaurant_user_id_dict

    def get_num_clients_per_restaurant(self) -> dict:
        num_clients_dict = defaultdict(list)
        for rest_id, clients_list in self.restaurant_clients_dict.items():
            num_clients_dict[len(clients_list)].append(rest_id)
        return num_clients_dict

    def __init__(self, items_df: pd.DataFrame, user_business_dict: dict, num_items_in_to_take: int, seed: int = 0):
        self.full_user_competition_business_ids = CompetitionFrame.calc_users_competition_restaurant(user_business_dict, num_items_in_to_take)
        self.users_recommendation_df_dict = None
        self.restaurant_clients_dict = None
        self.num_clients_dict = None
        self.user_competition_business_ids = None
        self.adjust_competition_frame_according_len_rec(items_df, num_items_in_to_take, seed)

    def adjust_competition_frame_according_len_rec(self, items_df: pd.DataFrame, num_items_in_list: int, seed: int):
        self.users_recommendation_df_dict, self.user_competition_business_ids = self.create_for_each_user_competition_data(items_df, num_items_in_list, seed)
        self.restaurant_clients_dict = self.get_restaurant_to_competitions_mapping()
        self.num_clients_dict = self.get_num_clients_per_restaurant()

    def save(self, output_path: str):
        with open(output_path, 'wb') as out_file:
            dill.dump(self, out_file)


def load_competition_frame(competition_path=competition_frame_restaurant_path) -> CompetitionFrame:
    with open(competition_path, "rb") as f:
        competition_frame = dill.load(f)
    return competition_frame


def run_competition_frame(items_df: pd.DataFrame):
    num_to_filter = num_rev_thresh_rest
    len_rec_list = len_rec_list_rest
    competition_path = competition_frame_restaurant_path
    user_business_dict = filter_user_few_rev(thresh=num_to_filter)
    competition_frame = CompetitionFrame(items_df, user_business_dict, len_rec_list)
    competition_frame.save(competition_path)


if __name__ == '__main__':
    data = pd.read_csv(restaurant_df_path)
    run_competition_frame(data)
