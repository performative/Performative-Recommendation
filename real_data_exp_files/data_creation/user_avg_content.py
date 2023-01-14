import pandas as pd
import numpy as np
import json
from collections import defaultdict
from real_data_exp_files.definitions import *


def filter_dup_user_business_rev(user_rev_dict):
    def cut_year_from_date(rev_element):
        return int(rev_element["date"].split('-')[0])

    user_business_dict = defaultdict(dict)
    for user_id, user_rev_list in user_rev_dict.items():
        for rev in user_rev_list:
            business_id = rev['business_id']
            if business_id in user_business_dict[user_id]:
                year_rev_dict = cut_year_from_date(user_business_dict[user_id][business_id])
                year_rev_current = cut_year_from_date(rev)
                if year_rev_current > year_rev_dict:
                    user_business_dict[user_id][business_id] = rev
            else:
                user_business_dict[user_id][business_id] = rev
    return user_business_dict


def filter_user_few_rev(thresh: float = 100) -> dict:
    path_to_load = user_rev_restaurant_path
    with open(path_to_load, 'r') as json_file:
        user_rev_dict = json.load(json_file)
    user_business_dict = filter_dup_user_business_rev(user_rev_dict)
    filtered_dict = dict()
    for user_id in user_business_dict:
        if len(user_business_dict[user_id]) >= thresh:
            filtered_dict[user_id] = user_business_dict[user_id]
    return filtered_dict


def create_avg_features(user_business_dict: dict, restaurant_df: pd.DataFrame, feature_dict_path: str):
    num_features = restaurant_df.shape[0]
    user_avg_features_dict = dict()
    for user_id, user_rev_dict in user_business_dict.items():
        user_avg_features_dict[user_id] = np.zeros(num_features)
        for restaurant_id in user_rev_dict.keys():
            user_avg_features_dict[user_id] += restaurant_df[restaurant_id]
        user_avg_features_dict[user_id] = ((user_avg_features_dict[user_id]/(len(user_rev_dict))).to_list())
    with open(feature_dict_path, 'w+') as json_file:
        json.dump(user_avg_features_dict, json_file, indent=4)

    user_features_df = pd.DataFrame.from_dict(user_avg_features_dict)
    user_features_df.to_csv(user_features_df_path)


def run_user_avg(num_to_filter: int, restaurant_df: pd.DataFrame, feature_dict_path: str):
    user_business_dict = filter_user_few_rev(num_to_filter)
    create_avg_features(user_business_dict, restaurant_df, feature_dict_path)


if __name__ == '__main__':
    items_df = pd.read_csv(restaurant_df_path)
    run_user_avg(num_rev_thresh_rest, items_df, user_features_df_path)
