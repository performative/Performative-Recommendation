import pandas as pd
import json
from collections import defaultdict
from tqdm import tqdm
from real_data_exp_files.definitions import restaurant_df_path, user_rev_restaurant_path, yelp_data


def create_rev_list(restaurants_data: pd.DataFrame):
    restaurants_id = set(restaurants_data.keys())
    user_rev_dict = defaultdict(list)  # user -> review
    with open(f'{yelp_data}/yelp_academic_dataset_review.json', 'r') as rev_f:
        for line in tqdm(rev_f):
            rev = json.loads(line)
            user_id = rev['user_id']
            business_id = rev['business_id']
            if business_id in restaurants_id:
                user_rev_dict[user_id].append(rev)
    with open(user_rev_restaurant_path, 'w+') as json_file:
        json.dump(user_rev_dict, json_file, indent=4)


if __name__ == '__main__':
    data = pd.read_csv(restaurant_df_path)
    create_rev_list(data)