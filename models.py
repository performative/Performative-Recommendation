import torch
import pandas as pd


class MLP(torch.nn.Module):
    def __init__(self, num_features: int, device: torch.cuda.device):
        super().__init__()
        torch.manual_seed(0)
        self.device = device
        layers = []
        layers.append(torch.nn.Linear(num_features, 2*num_features))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(2*num_features, num_features))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(num_features, int(num_features / 2)))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(int(num_features / 2), int(num_features / 4)))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(int(num_features / 4), 1))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, batch: torch.tensor) -> torch.tensor:
        return self.layers(batch.to(self.device))


class FStarWarraper(torch.nn.Module):
    def __init__(self, f_star_model: MLP):
        super().__init__()
        self.f_star = f_star_model

    def forward(self, batch: torch.tensor):
        return 2 ** (torch.nn.Sigmoid()(self.f_star(batch)))


class OneBitUserPreferenceEncoder(torch.nn.Module):
    def __init__(self, user_id_list: list, num_items_features: int, alpha: float = 0.1, seed: int = 8):
        super().__init__()
        torch.manual_seed(seed)
        layers = list()
        current_linear_input_size = len(user_id_list)
        layers.append(torch.nn.Linear(current_linear_input_size, num_items_features, bias=False))
        self.user_preference = torch.nn.Sequential(*layers)
        self.num_items_features = num_items_features
        self.user_id_list = user_id_list
        self.alpha = alpha
        self.one_hot_user_df = self.create_hot_bit_df()

    def get_user_preference(self, user_embed: torch.tensor, normalize: bool = True) -> torch.tensor:
        user_pref = self.user_preference(user_embed.float())
        if normalize:
            user_pref = torch.nn.functional.normalize(user_pref)
        return user_pref

    def create_hot_bit_df(self) -> pd.DataFrame:
        vector_dict = dict()
        for i, u_id in enumerate(self.user_id_list):
            vector_dict[u_id] = [0.0 for _ in range(len(self.user_id_list))]
            vector_dict[u_id][i - 1] = 1
        return pd.DataFrame.from_dict(vector_dict)

    def get_df_ids(self, id_list: list):
        return self.one_hot_user_df[id_list]

    def split_x_to_user_items_features(self, x: torch.tensor) -> (torch.tensor, torch.tensor):
        items_features, user_features = x[:, :, :self.num_items_features], x[:, 0, self.num_items_features:]
        return items_features, user_features

    def forward(self, x: torch.tensor):
        items_features, user_features = self.split_x_to_user_items_features(x)
        scores = items_features @ user_features.unsqueeze(2)
        return scores

    def get_items_after_strategic(self, item_features: torch.tensor, users_preference: torch.tensor) -> torch.tensor:
        avg_preference = users_preference.mean(dim=0) + 0.00001
        avg_preference = torch.nn.functional.normalize(avg_preference, dim=0)
        x_tag = torch.nn.functional.normalize(avg_preference + 2 * self.alpha * item_features, dim=0)
        return x_tag

