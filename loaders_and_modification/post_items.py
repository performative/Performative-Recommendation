import torch
import pandas as pd
from loaders_and_modification.strategic_data_loader import IterableStrategicYelpData


def get_post_items(iter_all_data: IterableStrategicYelpData, items_df: pd.DataFrame, saving_name: str = None) -> pd.DataFrame:
    iter_all_data.start_batch()
    copy_rest_df = items_df.copy()
    with torch.no_grad():
        post_items_df = iter_all_data.get_post_strategic_items_features(copy_rest_df)
    if saving_name:
        post_items_df.to_csv(saving_name, index=False)
    return post_items_df

