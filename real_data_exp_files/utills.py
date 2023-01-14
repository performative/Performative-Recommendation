import torch
import pandas as pd
from loaders_and_modification.strategic_data_loader import IterableStrategicYelpData
from loaders_and_modification.yelp_data_loader import IterableYelpData
from dataclasses import dataclass
from real_data_exp_files.data_creation.competition_frame import CompetitionFrame

@dataclass
class LambdaLineSearch:
    # high limit of binary line search
    lambda_high: float
    # low limit of binary line search
    lambda_low: float
    # number of tries to achieve target
    num_tries: int
    # the minimum improve in diversity to continue next try
    tol_div_imp: float
    # the max absolute between ndcg target to the ndcg model gets
    ndcg_target_thresh: float
    # the lowest lambda value to find in line search. if we get to lower value line search stops
    lambda_low_lim_to_stop: float
    # the highest lambda value to find in line search. if we get to higher value line search stops
    lambda_high_lim_to_stop: float


def from_float_to_str(float_: float):
    str_ = str(float_)
    if '.' in str_:
        str_before_dot = str_.split('.')[0]
        str_after_dot = str_.split('.')[1]
        str_ = str_before_dot + '_' + str_after_dot
    return str_


def get_data_loader(model: torch.nn.Module, user_ids: list, user_features_df: pd.DataFrame, f_star: torch.nn.Module,
                    device, current_items_df: pd.DataFrame, args, batch_size: int, competition_frame: CompetitionFrame,
                    train_mode: bool):
    if not train_mode or not args.strategic_model:
        return IterableYelpData(model, user_ids, user_features_df, f_star, device, args.batch_size, competition_frame)

    return IterableStrategicYelpData(model, user_ids, user_features_df, f_star, device, batch_size, current_items_df,
                                     competition_frame)
