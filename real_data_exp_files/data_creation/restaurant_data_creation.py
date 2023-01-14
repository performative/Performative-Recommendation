import json
import pandas as pd
from collections import defaultdict
import numpy as np
from real_data_exp_files.definitions import restaurant_df_all_feature_path, restaurant_df_col_path, yelp_data


def get_category_set(business_data_path: str) -> set:
    f = open(business_data_path, 'r')
    category_set = set()
    for line in f:
        data = json.loads(line)
        if data["categories"] is not None:
            category_list = data["categories"].replace(", ", ",").split(',')
            category_set = category_set.union(set(category_list))
    f.close()
    return category_set


def create_category_groups(category_set: set) -> (set,  set, set):
    must_be_categories = {'Burgers', 'Argentine', 'Austrian', 'Tex-Mex', 'Ethnic Food', 'Buffets', 'Poutineries',
                           'Italian', 'Macarons', 'Napoletana', 'Fish & Chips', 'Chicken Wings', 'Tacos', 'Guamanian',
                           'Canteen', 'Szechuan', 'Puerto Rican', 'Haitian', 'Pita', 'Chinese', 'Seafood', 'Mexican',
                           'Turkish', 'Fast Food', 'Latin American', 'Pancakes', 'Cheesesteaks', 'Singaporean', 'Belgian'
                           'Food Trucks', 'Waffles', 'British', 'Estheticians', 'Soul Food', 'Portuguese', 'Venezuelan',
                           'Spanish', 'Trinidadian', 'Ethiopian', 'Hot Dogs', 'Sri Lankan', 'Greek', 'Scandinavian',
                           'Empanadas', 'Comfort Food', 'Indian', 'Australian', 'Southern', 'Colombian', 'Noodles',
                           'Pizza', 'Japanese Curry', 'Armenian', 'New Mexican Cuisine', 'Filipino', 'Tapas Bars',
                           'Lebanese', 'Brasseries', 'Dominican', 'Malaysian', 'American (New)', 'Kebab', 'Diners',
                           'Swiss Food', 'Hungarian', 'Russian', 'Patisserie/Cake Shop', 'Salad', 'Somali', 'Dim Sum',
                           'German', 'Caterers', 'Falafel', 'ersian/Iranian', 'Cantonese', 'Nicaraguan', 'Sushi Bars',
                           'Ukrainian', 'Polynesian', 'Korean', 'Uzbek', 'Czech/Slovakian', 'Taiwanese', 'Congee',
                           'Moroccan', 'Shaved Ice', 'Izakaya', 'Pakistani', 'Burmese', 'South African', 'Bulgarian',
                           'Rotisserie Chicken', 'Halal', 'Peruvian', 'Bagels', 'Japanese', 'Vietnamese', 'Syrian',
                           'Egyptian', 'Arabian', 'Cupcakes', 'Czech', 'Georgian', 'Cambodian', 'Ayurveda', 'Cabinetry',
                           'Iberian', 'Sicilian', 'African', 'Canadian (New)', 'Hot Pot', 'Soup', 'Mongolian',
                           'Steakhouses', 'Sandwiches', 'Polish', 'Thai', 'Brazilian', 'Tacos', 'French', 'Indonesian'}

    may_be_categories = {'Cafeteria', 'Drive-Thru Bars', 'Irish Pub', 'Cheese Shops', 'Coffee & Tea',
                         'Pop-Up Restaurants', 'Specialty Food', 'Breweries', 'American (Traditional)', 'Beer Bar',
                         'Custom Cakes', 'Educational Services', 'Donuts', 'Restaurant Supplies', 'Cigar Bars',
                         'Fruits & Veggies', 'Pubs', 'Imported Food', 'Dance Clubs', 'Dinner Theater', 'Themed Cafes',
                         'Irish', 'Coffeeshops', 'Restaurants', 'Popcorn Shops', 'Beer', 'Scottish', 'Oriental',
                         'Jewish', 'Dance Restaurants', 'Bistros', 'Ice Cream & Frozen Yogurt', 'Brasseries',
                         'Internet Cafes', 'Kiosk', 'Wine Bars', 'Food Stands', 'Tea Rooms', 'Fishmonger', 'Cafes',
                         'Vermouth Bars', 'Food', 'Brewpubs', 'Gelato', 'Food Banks', 'Poke', 'Food Court',
                         'Bookstores', 'Fondue', 'Vegan', 'Hookah Bars', 'Juice Bars & Smoothies', 'Parent Cafes',
                         'Dive Bars', 'Whiskey Bars', 'Beer Garden', 'Hong Kong Style Cafe', 'Hawaiian', 'Sports Bars',
                         'Smokehouse', 'Bakeries', 'Kosher', 'Vegetarian', 'Gay Bars', 'Rugs', 'Coffee Roasteries',
                         'Desserts'}

    must_not_be_categories = category_set - may_be_categories.union(must_be_categories)
    return must_be_categories, must_not_be_categories, may_be_categories


def get_filter_business_to_restaurant(business_data_path: str) -> dict:
    category_set = get_category_set(business_data_path)
    must_be_categories, must_not_be_categories, may_be_categories = create_category_groups(category_set)
    f = open(business_data_path, 'r')
    restaurants_data = dict()
    for line in f:
        data = json.loads(line)
        if data["categories"] is not None:
            line_category_set = set(data["categories"].replace(", ", ",").split(','))
            if len(line_category_set.intersection(must_not_be_categories)) == 0 and len(
                    line_category_set.intersection(must_be_categories)) != 0:
                restaurants_data[data['business_id']] = data
    f.close()
    return restaurants_data


def create_special_category() -> dict:
    asian = {'Singaporean', 'Filipino', 'Dim Sum', 'Sushi Bars', 'Thai', 'Mongolian', 'Vietnamese', 'Cambodian',
             'Taiwanese', 'Korean', 'Burmese', 'Hot Pot'}
    japanese = {'Japanese', 'Japanese Curry'}
    chinese = {'Chinese'}
    india = {'Pakistani', 'Indian'}
    middle_east = {'Pita', 'Turkish', 'Falafel', 'ersian/Iranian', 'Pakistani', 'Syrian', 'Egyptian', 'Arabian',
                   'Lebanese', 'Kebab'}
    mexican_food = {'Mexican', 'Tacos', 'New Mexican Cuisine'}
    waffles = {'Waffles'}
    pancakes = {'Pancakes'}
    ice_cream = {'Ice Cream', 'Ice Cream & Frozen Yogurt'}
    other_sweets = {'Macarons', 'Donuts', 'Desserts', 'Cupcakes', 'Patisserie/Cake Shop', 'Custom Cakes'}
    coffe = {'Coffee & Tea', 'Themed Cafes', 'Parent Cafes', 'Coffee Roasteries', 'Coffeeshops', 'Internet Cafes',
             'Hong Kong Style Cafe'}
    italian = {'Sicilian', 'Italian', 'Napoletana'}
    latin_american = {'Brazilian', 'Empanadas', 'Argentine'}
    east_eu = {'Hungarian', 'Russian', 'Polish', 'Ukrainian'}
    Burgers = {'Burgers'}
    middle_eu = {'Austrian', 'Bulgarian', 'Swiss Food'}
    Hot_Dogs = {'Hot Dogs'}
    Sandwiches = {'Sandwiches'}
    steak = {'Cheesesteaks', 'Steakhouses', 'Smokehouse'}
    Pizza = {'Pizza'}
    Seafood = {'Seafood'}
    fast_food = {'Fast Food'}
    vegan = {'Salad', 'Vegetarian', 'Vegan'}
    french = {'French'}
    special_category_dict = {'asian': asian, 'japanese': japanese, 'chinese': chinese, 'india': india,
                             'middle_east': middle_east, 'mexican_food': mexican_food, 'sweets': other_sweets,
                             'coffe': coffe, 'italian': italian, 'latin_american': latin_american, 'east_eu': east_eu,
                             'Burgers': Burgers, 'middle_eu': middle_eu, 'Hot_Dogs': Hot_Dogs, 'Sandwiches': Sandwiches,
                             'steak': steak, 'Pizza': Pizza, 'Seafood': Seafood, 'fast_food': fast_food, 'vegan': vegan,
                             'french': french, 'waffles': waffles, 'pancakes': pancakes, 'ice_cream': ice_cream}
    return special_category_dict


def get_flat_rest_dict(restaurants_data: dict, special_category_dict: dict) -> dict:
    flatten_dict = defaultdict(dict)
    for id, restaurant in restaurants_data.items():
        for key in restaurant.keys():
            if type(restaurant[key]) is dict:
                for att_key, att in restaurant[key].items():
                    if type(att) is str and att.startswith('{'):
                        if att == '{}':
                            continue
                        for att_option_tup in att[1:-1].replace(' ', '').split(','):
                            att_option_key, att_option_value = att_option_tup.split(':')
                            att_option_key = att_option_key[1:-1]
                            flatten_dict[id][att_key + '_' + att_option_key] = att_option_value
                    else:
                        flatten_dict[id][att_key] = att
            else:
                if key == "categories":
                    restaurant_categories = set(restaurant["categories"].replace(", ", ",").split(','))
                    for food_subcategory_set in special_category_dict:
                        if len(restaurant_categories.intersection(special_category_dict[food_subcategory_set])) > 0:
                            flatten_dict[id][food_subcategory_set] = 1
                        else:
                            flatten_dict[id][food_subcategory_set] = -1
                else:
                    flatten_dict[id][key] = restaurant[key]
    return flatten_dict


def from_restaurants_dict_to_df(restaurants_data: dict):
    special_category_dict = create_special_category()
    rest_flatten_dict = get_flat_rest_dict(restaurants_data, special_category_dict)
    restaurant_df = pd.DataFrame.from_dict(rest_flatten_dict).T
    return restaurant_df


def norm_column(col_name: str, df: pd.DataFrame):
    mean = df[col_name].mean()
    std = df[col_name].std()
    df[col_name] = (df[col_name] - mean) / std


def normalize_features(restaurant_df: pd.DataFrame):
    restaurant_df['review_count'] = np.log(np.log(restaurant_df['review_count'].astype(float)))
    restaurant_df.rename(columns={'review_count': 'log_log_review_count'}, inplace=True)
    norm_column('log_log_review_count', restaurant_df)
    norm_column('stars', restaurant_df)
    norm_column('RestaurantsPriceRange2', restaurant_df)


def map_true_false(df: pd.DataFrame, col_name: str):
    return df[col_name].map({
        'True': 1,
        'False': -1,
        'None': -1
    }).fillna(-1)


def handle_na_and_words(restaurant_df: pd.DataFrame):
    restaurant_df['RestaurantsTableService'] = restaurant_df['RestaurantsTableService'].fillna('False').replace('None',
                                                                                                                'False')
    restaurant_df['RestaurantsTableService'] = restaurant_df['RestaurantsTableService'].replace('False', 0).replace(
        'True', 1)

    binary_column = ['BikeParking', 'BusinessParking_garage', 'BusinessParking_street', 'BusinessParking_validated',
                     'BusinessParking_lot', 'BusinessParking_valet', 'BusinessAcceptsCreditCards',
                     'RestaurantsReservations', 'WheelchairAccessible', 'Caters', 'OutdoorSeating',
                     'RestaurantsGoodForGroups', 'HappyHour', 'Ambience_touristy', 'Ambience_hipster',
                     'Ambience_romantic', 'Ambience_divey', 'Ambience_intimate', 'Ambience_trendy', 'Ambience_upscale',
                     'Ambience_classy', 'Ambience_casual', 'HasTV', 'GoodForMeal_dessert', 'GoodForMeal_latenight',
                     'GoodForMeal_lunch', 'GoodForMeal_dinner', 'GoodForMeal_brunch', 'GoodForMeal_breakfast',
                     'DogsAllowed', 'RestaurantsTakeOut', 'RestaurantsDelivery', 'GoodForKids', 'Music_dj',
                     'Music_background_music', 'Music_jukebox', 'Music_live', 'Music_video', 'Music_karaoke',
                     'GoodForDancing', 'DriveThru', 'Corkage']

    for col in binary_column:
        restaurant_df[col] = map_true_false(restaurant_df, col)
    price_range_avg = restaurant_df['RestaurantsPriceRange2'].dropna().replace('None', '-1').astype(float).mean()
    restaurant_df['RestaurantsPriceRange2'] = restaurant_df['RestaurantsPriceRange2'].fillna(price_range_avg).replace(
        'None', price_range_avg).astype(float)

    restaurant_df['Alcohol'] = restaurant_df['Alcohol'].map({
        'u\'full_bar\'': 1,
        'u\'none\'': -1,
        '\'full_bar\'': 1,
        '\'none\'': -1,
        'u\'beer_and_wine\'': 0.0,
        '\'beer_and_wine\'': 0.0,
        'None': -1
    }).fillna(-1)

    restaurant_df['NoiseLevel'] = restaurant_df['NoiseLevel'].map({
        'u\'average\'': 0.0,
        'u\'quiet\'': -1,
        '\'average\'': 0.0,
        'u\'loud\'': 0.5,
        '\'quiet\'': -1,
        'u\'very_loud\'': 1,
        '\'loud\'': 0.5,
        '\'very_loud\'': 1,
        'None': 0.0
    }).fillna(-1)

    restaurant_df['RestaurantsAttire'] = restaurant_df['RestaurantsAttire'].map({
        'u\'casual\'': -1,
        '\'casual\'': -1,
        'u\'dressy\'': 0.0,
        '\'dressy\'': 0.0,
        'u\'formal\'': 1,
        '\'formal\'': 1
    }).fillna(-1)

    restaurant_df['WiFi'] = restaurant_df['WiFi'].map({
        'u\'free\'': 1,
        'u\'no\'': -1,
        '\'free\'': 1,
        '\'no\'': -1,
        'u\'paid\'': 0.0,
        '\'paid\'': 0.0,
        'None': -1
    }).fillna(-1)

    for special_category in create_special_category().keys():
        restaurant_df[special_category].fillna(-1, inplace=True)


def drop_unnecessary_features(restaurant_df: pd.DataFrame):
    feature_to_drop = ['BusinessAcceptsBitcoin', 'ByAppointmentOnly', 'attributes', 'BestNights_monday',
                       'BestNights_tuesday', 'BestNights_friday', 'BestNights_wednesday', 'BestNights_thursday',
                       'BestNights_sunday', 'BestNights_saturday', 'BYOB', 'CoatCheck', 'Smoking', 'BYOBCorkage',
                       'RestaurantsCounterService', 'BusinessParking', 'Ambience', 'GoodForMeal', 'Open24Hours',
                       'business_id', 'name', 'address', 'city', 'state', 'postal_code', 'latitude', 'longitude',
                       'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'is_open',
                       'DietaryRestrictions_dairy-free', 'DietaryRestrictions_gluten-free', 'DietaryRestrictions_vegan',
                       'DietaryRestrictions_kosher', 'DietaryRestrictions_halal', 'DietaryRestrictions_soy-free',
                       'DietaryRestrictions_vegetarian', 'Music_no_music', 'hours', 'Ambience_romantic',
                       'Ambience_intimate', 'Ambience_classy', 'Ambience_hipster', 'Ambience_divey',
                       'Ambience_touristy', 'Ambience_trendy', 'Ambience_upscale', 'Ambience_casual', 'WiFi',
                       'latin_american', 'east_eu', 'middle_eu', 'french', 'waffles', 'pancakes', 'Music_dj',
                       'Music_background_music', 'Music_no_music', 'Music_jukebox', 'Music_live', 'Music_video',
                       'Music_karaoke', 'GoodForDancing', 'BusinessParking_garage', 'BusinessParking_street',
                       'BusinessParking_validated', 'BusinessParking_lot', 'BusinessParking_valet', 'asian',
                       'log_log_review_count', 'GoodForKids',
                       ]
    restaurant_df.drop(columns=feature_to_drop, inplace=True)


def create_restaurant_data():
    business_data_path = f'{yelp_data}/yelp_academic_dataset_business.json'
    restaurants_data = get_filter_business_to_restaurant(business_data_path)
    restaurant_df = from_restaurants_dict_to_df(restaurants_data)
    handle_na_and_words(restaurant_df)
    normalize_features(restaurant_df)
    restaurant_df.to_csv(restaurant_df_all_feature_path, index_label=[restaurant_df.index.name])
    drop_unnecessary_features(restaurant_df)
    restaurant_df.to_csv(restaurant_df_col_path, index=True)


if __name__ == '__main__':
    create_restaurant_data()