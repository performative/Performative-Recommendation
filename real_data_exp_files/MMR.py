import sys
sys.path.append("..")
from real_data_exp_files.utills import *
from real_data_exp_files.data_creation.competition_frame import *
from argparse import ArgumentParser
from models import OneBitUserPreferenceEncoder, FStarWarraper
import pandas as pd
import torch
from ndcg_div.disimilarity_metrics import dissimilarity_cosine
from ndcg_div.ndcg_div_loss import Loss
from real_data_exp_files.f_star_code.f_star_creator import get_f_star


def mmr_parse_args():
    parser = ArgumentParser()
    parser.add_argument("--c", dest="alpha", type=float, required=False, default=0.1)
    parser.add_argument("--r", dest="reg", type=float, required=False, default=2/3)
    parser.add_argument("--k", dest="k", type=float, required=False, default=10)
    return parser.parse_args()


def calc_div(items_feature_tensor: torch.tensor, device: torch.cuda.device):
    num_items = items_feature_tensor.shape[1]
    dissimilarity_matrix = dissimilarity_cosine(items_feature_tensor.T.unsqueeze(0), device)
    clean_dissimilarity_matrix = Loss.zero_below_diagonal(dissimilarity_matrix)
    div_sum = clean_dissimilarity_matrix.sum().cpu().item()
    div = (2 * div_sum) / (num_items * (num_items-1))
    return div


def calc_dcg(relevance: torch.tensor, device: torch.cuda.device):
    discounted_rel = torch.log2(torch.arange(2, relevance.shape[0] + 2).float().to(device))
    dcg = ((2 ** relevance - 1) / discounted_rel).sum(axis=0)
    return dcg.item()


def find_next_item(items_id_set: set[str], chosen_items: set[str], current_df: pd.DataFrame, device: torch.cuda.device,
                   max_score: torch.tensor, user_pref: torch.tensor, reg: float):
    max_div_rel_score, item_to_add_id = None, None
    for candidate_item_id in items_id_set:
        features_with_candidate = current_df[list(chosen_items) + [candidate_item_id]]
        features_with_candidate_tensor = torch.from_numpy(features_with_candidate.to_numpy()).to(device).float()
        div_with_candidate = calc_div(features_with_candidate_tensor, device)
        candidate_features = torch.from_numpy(current_df[candidate_item_id].to_numpy()).float().to(device)
        candidate_rel_score = candidate_features @ user_pref
        candidate_score = reg * div_with_candidate + ((1 - reg) * candidate_rel_score / max_score).item()
        if max_div_rel_score is None or candidate_score > max_div_rel_score:
            max_div_rel_score = candidate_score
            item_to_add_id = candidate_item_id
    return item_to_add_id


def chose_item_to_user_and_get_best_dcg(k: int, items_id_set: set[str], current_df: pd.DataFrame, user_pref: torch.tensor,
                                        f_star: FStarWarraper, device: torch.cuda.device, user_feature: torch.tensor,
                                        reg: float) -> (set[str], float):
    chosen_items = set()
    max_score = None
    best_dcg = None
    while len(chosen_items) < k:
        items_id_list = list(items_id_set)
        items_features = torch.from_numpy(current_df[items_id_list].to_numpy()).to(device).float()
        if len(chosen_items) == 0:
            scores = items_features.T @ user_pref
            relevance = f_star(torch.cat((items_features.T, user_feature.repeat(items_features.shape[1], 1)), dim=1))
            best_dcg = calc_dcg(relevance.squeeze(-1).sort(descending=True)[0][:k], device)
            item_index_in_list = torch.argmax(scores).item()
            max_score = torch.max(scores)
            item_to_add_id = items_id_list[item_index_in_list]
        else:
            item_to_add_id = find_next_item(items_id_set, chosen_items, current_df, device, max_score, user_pref, reg)
        chosen_items.add(item_to_add_id)
        items_id_set.remove(item_to_add_id)
    return chosen_items, best_dcg


def get_ndcg_div_in_time(all_user_pref: torch.tensor, user_hot_bit_df: pd.DataFrame, user_features_df: pd.DataFrame,
                         device: torch.tensor, competition_frame_full: CompetitionFrame, current_df: pd.DataFrame,
                         f_star: FStarWarraper, reg: float, k: int):
    div_per_user_list = list()
    ndcg_per_user_list = list()
    for user_index, u_id in enumerate(user_hot_bit_df.columns):
        user_pref = all_user_pref[user_index]
        user_feature = torch.from_numpy(user_features_df[u_id].to_numpy()).to(device).float()
        items_id_set = set(competition_frame_full.full_user_competition_business_ids[u_id])
        chosen_items, best_dcg = chose_item_to_user_and_get_best_dcg(k, items_id_set, current_df, user_pref, f_star,
                                                                     device, user_feature, reg)
        selected_items_features = torch.from_numpy(current_df[list(chosen_items)].to_numpy()).float().to(device)
        relevance = f_star(
            torch.cat((selected_items_features.T, user_feature.repeat(selected_items_features.shape[1], 1)), dim=1))
        dcg = calc_dcg(relevance.squeeze(-1), device)
        ndcg = dcg / best_dcg
        ndcg_per_user_list.append(ndcg)
        diversity_mmr_top_k = calc_div(selected_items_features, device)
        div_per_user_list.append(diversity_mmr_top_k)
    div_time = sum(div_per_user_list) / len(div_per_user_list)
    ndcg_time = sum(ndcg_per_user_list) / len(ndcg_per_user_list)
    return ndcg_time, div_time


def wrapper_mmr():
    args = mmr_parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    competition_frame_full = load_competition_frame()
    with open(user_features_dict_path, 'r') as json_file:
        user_feature_dict = json.load(json_file)
    users_ids_list = list(user_feature_dict.keys())
    user_features_df = pd.read_csv(user_features_df_path).drop(columns=['Unnamed: 0'])
    current_df = pd.read_csv(restaurant_df_path)
    f_star = FStarWarraper(get_f_star(current_df, force_to_train=False)).float().to(device)
    model = OneBitUserPreferenceEncoder(users_ids_list, current_df.shape[0], args.alpha).to(device)
    model.eval()
    times_ndcg_list = []
    times_div_list = []
    for t in range(10):
        if t > 0:
            data_path = f'{strategic_data_folder}/k_10_{from_float_to_str(args.alpha)}/2_0/non_strategic_model/data_time_{t}.csv'
            print(f'load: {data_path}')
            current_df = pd.read_csv(data_path)
            competition_frame_full.adjust_competition_frame_according_len_rec(current_df, len_rec_list_rest, seed=t)
        model_path = f'{models_yelp}/k_10_{from_float_to_str(args.alpha)}/2_0/non_strategic_model/model_time_{t}.pt'
        model.load_state_dict(torch.load(model_path))
        user_hot_bit_df = model.get_df_ids(users_ids_list)
        user_hot_bit_tensor = torch.from_numpy(user_hot_bit_df.to_numpy()).float().to(device)
        with torch.no_grad():
            all_user_pref = model.user_preference(user_hot_bit_tensor)
            ndcg_time, div_time = get_ndcg_div_in_time(all_user_pref, user_hot_bit_df, user_features_df,  device,
                                                       competition_frame_full, current_df,  f_star, args.reg, args.k)
        print(f'time: {t} ndcg: {ndcg_time} div: {div_time}', flush=True)
        times_div_list.append(div_time)
        times_ndcg_list.append(ndcg_time)
    print('all ndcg:')
    print(times_ndcg_list)
    print('all div:')
    print(times_div_list)


if __name__ == '__main__':
    wrapper_mmr()
