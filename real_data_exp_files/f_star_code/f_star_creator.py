import random
import pandas as pd
import torch
import os
import sys
import json
from models import MLP
import torch.optim as optim
from real_data_exp_files.f_star_code.f_star_trainer import FStarTrainer
from real_data_exp_files.data_creation.user_avg_content import filter_user_few_rev
from real_data_exp_files.definitions import *
from tqdm import tqdm
sys.path.append('.')


def split_train_val(all_data_list: list[dict]) -> (list[dict], list[dict], list[dict]):
    '''

    :param all_data_list: List of  all reviews for all users
    :return: Partition of all_data_list to three different list train, validation and test.
    '''
    random.seed(8)
    random.shuffle(all_data_list)
    train_val_splitter, val_test_splitter = int(0.7 * len(all_data_list)), int(0.9 * len(all_data_list))
    train_list, validation_list, test_list = all_data_list[:train_val_splitter], all_data_list[train_val_splitter:val_test_splitter], \
                                             all_data_list[val_test_splitter:]
    return train_list, validation_list, test_list


def get_filtered_rev_list(user_business_filtered_dict: dict) -> list[dict]:
    '''
    :param user_business_filtered_dict: A dictionary that each key is user id and value is sub dictionary with business
     id as keys and review data as items without duplicates that user has more than one review on items
    :return: cast the dictionary to review list.
    '''
    rev_list = list()
    for user_id in user_business_filtered_dict:
        for rev_data in user_business_filtered_dict[user_id].values():
            rev_list.append(rev_data)
    return rev_list


def from_rev_list_to_batch_generator(rev_list: list[dict], restaurant_df: pd.DataFrame, user_abs_dict: dict, batch_size=256):
    '''
    :param rev_list: list of all the reviews
    :param restaurant_df: df that includes all features of all restaurant. Each column is a different restaurant
    :param user_abs_dict: A dictionary that contains for each user his abstract features.
    :param batch_size: The size of batch to generate
    :return: Each call returns batch i.e tensors of items and user features features in the shape of batch_size, num_features
    and true labels tensor in shape of batch_size
    '''
    random.shuffle(rev_list)
    for i in range(0, len(rev_list), batch_size):
        X, y = list(), list()
        for rev in rev_list[i:i + batch_size]:
            business_id = rev['business_id']
            user_id = rev['user_id']
            y.append(rev['stars'])
            user_features = torch.tensor(user_abs_dict[user_id])
            restaurant_features = torch.tensor(restaurant_df[business_id].to_numpy()).float()
            X.append(torch.cat((restaurant_features, user_features)))
        X = torch.stack(X)
        y = torch.tensor(y)
        yield X, y
        i += 1


def get_rev_list(user_business_dict: dict) -> list[dict]:
    '''

    :param user_business_dict:  A dictionary that the keys are user ids and the values are list of the restaurant they
     reviewed.
    :return: Review list
    '''
    print('create rev list')
    rest_all_f = pd.read_csv(restaurant_all_features_df_path)
    rev_list = list()
    print(len(user_business_dict.keys()), flush=True)
    for u_id in tqdm(user_business_dict.keys()):
        u_rest = set()
        for business_id, rev in user_business_dict[u_id].items():
            assert business_id == rev['business_id']
            assert u_id == rev['user_id']
            u_rest.add(business_id)
            rev_list.append({'business_id': business_id, 'user_id': u_id, 'stars': 1.0})
        u_rest_list = list(u_rest)
        for rest in u_rest_list:
            latitude, longitude = tuple(rest_all_f[rest_all_f['business_id'] == rest][['latitude', 'longitude']].values[0].tolist())
            dist_to_center = (rest_all_f['latitude'] - latitude) ** 2 + (rest_all_f['longitude'] - longitude) ** 2
            dist_df = pd.DataFrame({'id': rest_all_f['business_id'], 'dist_to_center': dist_to_center})
            closest_rest = dist_df.sort_values(by=['dist_to_center'])['id'].to_list()
            for close_rest in closest_rest:
                if close_rest not in u_rest:
                    u_rest.add(close_rest)
                    rev_list.append({'business_id': close_rest, 'user_id': u_id, 'stars': 0.0})
                    break
    with open(rev_list_path, 'w+') as json_file:
        json.dump(rev_list, json_file, indent=4)
    print('finish create list of revs..')
    return rev_list


def get_f_star(restaurant_df: pd.DataFrame, force_to_train: bool = True, create_rev_list: bool = False) -> MLP:
    '''

    :param restaurant_df:  df that includes all features of all restaurant. Each column is a different restaurant
    :param force_to_train: Force to train f* model
    :param create_rev_list: Force to create new review list. Only relevant if force_to_train is True
    :return: A trained f* model
    '''
    with open(user_features_f_star_df_path, 'r') as json_file:
        user_features_dict = json.load(json_file)
    num_items_features = restaurant_df.shape[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    f_star = MLP(num_items_features * 2, device)
    if force_to_train:
        print('training f*')
        user_business_dict = filter_user_few_rev(thresh=num_rev_thresh_rest_f_star)
        if create_rev_list or not os.path.exists(rev_list_path):
            rev_list = get_rev_list(user_business_dict)
        else:
            with open(rev_list_path) as json_file:
                rev_list = json.load(json_file)

        train_list, validation_list, test_list = split_train_val(rev_list)
        train_generator_creator = lambda: from_rev_list_to_batch_generator(train_list, restaurant_df, user_features_dict)
        val_generator_creator = lambda: from_rev_list_to_batch_generator(validation_list, restaurant_df, user_features_dict)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        learn_rate = 0.01
        betas = (0.9, 0.999)
        optimizer = optim.Adam(f_star.parameters(), lr=learn_rate, betas=betas)
        trainer = FStarTrainer(f_star, optimizer, device)
        best_model_state, best_res, best_epoch = trainer.fit(train_generator_creator, val_generator_creator,
                                                             num_epoch=120, print_every=1)
        os.makedirs(models_yelp, exist_ok=True)
        torch.save(best_model_state, f_star_path)
        f_star.load_state_dict(torch.load(f_star_path))
        test_generator_creator = lambda: from_rev_list_to_batch_generator(test_list, restaurant_df, user_features_dict)
        trainer.eval_on_test(test_generator_creator)
    else:
        f_star.load_state_dict(torch.load(f_star_path, map_location=device))
    return f_star


if __name__ == '__main__':
    items_df = pd.read_csv(restaurant_df_path)
    items_df = items_df / ((items_df ** 2).sum(axis=0) ** 0.5)
    get_f_star(items_df, force_to_train=True)
