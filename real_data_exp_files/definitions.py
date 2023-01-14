models_yelp = 'models_yelp'
result_folder = 'result'
plots_path = f'{result_folder}/plots'
yelp_data = 'yelp_data'
f_star_path = f'{models_yelp}/f_star.pt'
restaurant_df_all_feature_path = f'{yelp_data}/restaurant_features.csv'
restaurant_df_path = f'{yelp_data}/restaurant_features_for_pred.csv'
restaurant_df_col_path = f'{yelp_data}/restaurant_features_for_pred_col.csv'

user_features_dict_path = f'{yelp_data}/user_content_avg_features.json'
user_features_df_path = f'{yelp_data}/user_content_avg_features.csv'
user_features_f_star_df_path = f'{yelp_data}/user_content_avg_features_f_star.csv'
user_rev_restaurant_path = f'{yelp_data}/users_review_restaurant.json'

restaurant_all_features_df_path = f'{yelp_data}/restaurant_features.csv'
competition_frame_restaurant_path = f"{yelp_data}/competition_frame.pkl"
strategic_data_folder = f'{yelp_data}/strategic_data'

rev_list_path = f'{yelp_data}/rev_list.json'
num_rev_thresh_rest = 100
num_rev_thresh_rest_f_star = 50
len_rec_list_rest = 40
len_rec_list_train = 20
len_rec_list_val = 30
len_rec_list_test = 40

rest_features = [ 'stars', 'Alcohol',
       'RestaurantsGoodForGroups', 'RestaurantsReservations',
       'RestaurantsAttire', 'BikeParking', 'RestaurantsPriceRange2', 'HasTV',
       'NoiseLevel', 'RestaurantsTakeOut', 'Caters', 'OutdoorSeating',
       'GoodForMeal_dessert', 'GoodForMeal_latenight', 'GoodForMeal_lunch',
       'GoodForMeal_dinner', 'GoodForMeal_brunch', 'GoodForMeal_breakfast',
       'DogsAllowed', 'RestaurantsDelivery', 'japanese', 'chinese',
       'india', 'middle_east', 'mexican_food', 'sweets', 'coffe', 'italian',
       'Burgers', 'Hot_Dogs', 'Sandwiches', 'steak', 'Pizza', 'Seafood',
       'fast_food', 'vegan', 'ice_cream', 'RestaurantsTableService',
       'BusinessAcceptsCreditCards', 'WheelchairAccessible', 'DriveThru',
       'HappyHour', 'Corkage']


