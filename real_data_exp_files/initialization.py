import sys
sys.path.append("..")
from real_data_exp_files.definitions import *
from real_data_exp_files.data_creation.user_avg_content import run_user_avg
from real_data_exp_files.data_creation.competition_frame import run_competition_frame
import pandas as pd
import os
from real_data_exp_files.f_star_code.f_star_creator import get_f_star
from real_data_exp_files.data_creation.user_review_creation import create_rev_list
from real_data_exp_files.data_creation.restaurant_data_creation import create_restaurant_data


def main():
    print('start', flush=True)
    os.makedirs(models_yelp, exist_ok=True)
    # create_restaurant_data has been used to extract the relevant features from the original yelp data. However the full data is not avaialbel so this function
    # cannot executed it can only be browsed. However the output files are in real_data_exp_files/yelp_data.
    # create_restaurant_data() 

    print('read_data', flush=True)

    data = pd.read_csv(restaurant_all_features_df_path, index_col=0)[rest_features]
    data.to_csv(restaurant_df_col_path, index=True)
    data = data.T
    data = data / ((data ** 2).sum(axis=0) ** 0.5)
    print(restaurant_df_path, flush=True)
    data.to_csv(restaurant_df_path, index=False)

    # create_rev_list is It's designed to match users with restaurants they've reviewed. However the full data is not avaialbel so this function
    # cannot executed it can only be browsed. However the output file is Compressed as a zip file in real_data_exp_files/yelp_data. You have to extract him before
    # executing this script.
    # create_rev_list(data)

    print('user avg', flush=True)
    run_user_avg(num_rev_thresh_rest_f_star, data, user_features_f_star_df_path)
    print('f*', flush=True)

    get_f_star(data, force_to_train=True)
    run_user_avg(num_rev_thresh_rest, data, user_features_df_path)
    run_competition_frame(data)


if __name__ == '__main__':
    main()