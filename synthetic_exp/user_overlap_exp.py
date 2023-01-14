import sys
sys.path.append("..")
import random
import torch
import numpy as np
from loaders_and_modification.strategic_data_loader import IterableStrategicYelpData
from models import OneBitUserPreferenceEncoder
from ndcg_div.ndcg_div_trainer import NdcgDivTrainer
import torch.optim as optim
from ndcg_div.ndcg_div_loss import Loss
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from synthetic_exp.user_dispersal_exp import FStartSynthetic, create_random_df
from synthetic_exp.synthetic_exp_utills import CompetitionFrameFromDict


def scatter_move_dots(items_df, post_items, pref_df, title=None, path=None):
    if title:
        plt.title(title)
    plt.ylim(-1.1, 1.1)
    plt.xlim(-1.1, 1.1)
    plt.scatter(post_items.loc[0], post_items.loc[1], color='red')
    plt.scatter(0, 0)
    plt.annotate('o', (0, 0))
    for item_name in items_df.columns:
        plt.plot([0, post_items[item_name][0]], [0, post_items[item_name][1]], color='black')
        plt.annotate(item_name + '\'', (post_items[item_name][0], post_items[item_name][1]))
    for u_id in pref_df.columns:
        plt.plot([0, pref_df[u_id][0]], [0, pref_df[u_id][1]], color='purple')
        plt.annotate(u_id, (pref_df[u_id][0], pref_df[u_id][1]))
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.clf()


def run_exp(args):
    num_dim = args.dim
    alpha = 0.0
    f_star = FStartSynthetic(num_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    div_list = list()
    ndcg_list = list()
    for iteration in range(args.num_iters):
        random.seed(iteration)
        np.random.seed(iteration)
        user_items_dict, items_list = get_connection_graph(args.num_users, args.num_items, args.num_in_list)
        user_items_dict = change_graph(user_items_dict, args.num_swap, args.num_in_list)

        user_id_list = list(user_items_dict.keys())
        num_users = len(user_id_list)
        k = len(user_items_dict[user_id_list[0]])
        batch_size = num_users

        u_star = np.ones(num_dim) / np.sqrt(num_dim)
        items_df = create_random_df(items_list, num_dim, std=args.x_std, p_star=u_star)
        user_df = create_random_df(list(user_id_list), num_dim, std=args.u_std, p_star=u_star)
        competition_frame = CompetitionFrameFromDict(user_items_dict, items_df)
        model = OneBitUserPreferenceEncoder(user_id_list, num_dim, alpha=alpha, seed=iteration)
        optimizer = optim.Adam(model.parameters(), lr=args.learn_rate, betas=(0.9, 0.999))
        loss = Loss(num_dim, top_k=k, sig_tau=args.tau_sig, tau_ndcg=args.tau_ndcg, tau_div=args.tau_div)
        iter_data_train = IterableStrategicYelpData(model, user_id_list, user_df, f_star, device,
                                                    batch_size,
                                                    items_df, competition_frame=competition_frame)

        trainer = NdcgDivTrainer(model, loss, optimizer, device, f_star, lambda_=args.lambda_, batch_size=batch_size)

        ndcg_score_, div_score_, _ = trainer.fit(iter_data_train, args.num_epoch,
                                                 print_every=1, get_best=True)
        div_list.append(div_score_)
        ndcg_list.append(ndcg_score_)
        print(f'iter: {iteration} ndcg: {ndcg_score_} div: {div_score_}', flush=True)

        iter_data_train.start_batch()
        post_items = iter_data_train.get_post_strategic_items_features(items_df.copy())
        model_created_pref = model.get_user_preference(
            torch.from_numpy(model.create_hot_bit_df().to_numpy()).to(device).T)
        model_created_pref = pd.DataFrame(model_created_pref.T.detach().cpu().numpy(), columns=user_id_list)
        scatter_move_dots(items_df, post_items, model_created_pref, title=f'div: {div_score_}')

    ndcg_array = np.array(ndcg_list)
    div_array = np.array(div_list)
    ndcg_mean, ndcg_std = ndcg_array.mean(), ndcg_array.std()
    div_mean, div_std = div_array.mean(), div_array.std()

    print(f"number changes: {args.num_swap}", flush=True)
    print(f'ndcg mean: {ndcg_mean} ndcg std: {ndcg_std} div mean: {div_mean} div std: {div_std}', flush=True)


def get_connection_graph(num_users: int, num_items, num_in_list: int):
    # number of users in round (num clicks) = x/l. number of rounds u*l/x. click size = num_rounds
    assert num_items % num_in_list == 0, "number of items should be integer multiplication of the number of List length"
    items_id_list = [f'x{i}' for i in range(1, num_items+1)]
    user_items_dict = dict()
    # click_size = num_users * num_in_list / num_items
    t = 1
    for i in range(int(num_users * num_in_list/num_items)):
        for j in range(int(num_items / num_in_list)):
            user_items_dict[f'u{t}'] = items_id_list[j*num_in_list:(j+1)*num_in_list]
            t += 1
    return user_items_dict, items_id_list


def change_graph(user_item_dicts: dict, num_swap: int, num_in_list):
    u_id_list = list(user_item_dicts.keys())
    while num_swap > 0:
        random.shuffle(u_id_list)
        u1, u2 = tuple(u_id_list[:2])
        u1_item_index = random.randint(0, num_in_list-1)
        u2_item_index = random.randint(0, num_in_list-1)
        if (user_item_dicts[u1][u1_item_index] not in set(user_item_dicts[u2])) and (user_item_dicts[u2][u2_item_index] not in set(user_item_dicts[u1])):
            tmp = user_item_dicts[u2][u2_item_index]
            user_item_dicts[u2][u2_item_index] = user_item_dicts[u1][u1_item_index]
            user_item_dicts[u1][u1_item_index] = tmp
            num_swap -= 1
    return user_item_dicts


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--list", dest='num_in_list', type=int, required=False, default=10)
    parser.add_argument("--nu", dest='num_users', type=int, required=False, default=200)
    parser.add_argument("--ni", dest='num_items', type=int, required=False, default=50)
    parser.add_argument("-ne", dest='num_epoch', type=int, required=False, default=500)
    parser.add_argument("--ustd", dest='u_std', type=float, required=False, default=0.1)
    parser.add_argument("--xstd", dest='x_std', type=float, required=False, default=1)
    parser.add_argument("--lambda", dest='lambda_', type=float, required=False, default=1000)
    parser.add_argument("--tau_sig", dest="tau_sig", type=float, required=False, default=2)
    parser.add_argument("--tau_div", dest="tau_div", type=float, required=False, default=5)
    parser.add_argument("--tau_ndcg", dest="tau_ndcg", type=float, required=False, default=5)
    parser.add_argument("--lr", dest="learn_rate", type=float, required=False, default=0.01)
    parser.add_argument("--dim", dest='dim', type=int, required=False, default=2)
    parser.add_argument("--iter", dest='num_iters', type=int, required=False, default=10)
    parser.add_argument("--num_swap", dest='num_swap', type=int, required=False, default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    run_exp(args)







