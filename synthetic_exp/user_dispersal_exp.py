import sys
sys.path.append("..")
from loaders_and_modification.strategic_data_loader import IterableStrategicYelpData
from loaders_and_modification.yelp_data_loader import IterableYelpData
from ndcg_div.ndcg_div_trainer import NdcgDivTrainer
from ndcg_div.ndcg_div_loss import Loss
import torch.optim as optim
from models import OneBitUserPreferenceEncoder
from argparse import ArgumentParser
from synthetic_exp.synthetic_exp_utills import *


def from_numbers_to_id(prefix: str, length: int):
    return [f'{prefix}{i}' for i in range(1, length+1)]


def run_exp_on_random_graph(args, k: int, seed: int) -> (float, float):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    items_id_list = from_numbers_to_id('x', args.num_items)
    u_star = np.ones(args.dim) / np.sqrt(args.dim)
    user_id_list = from_numbers_to_id('u', args.num_users)
    pref_df, items_df, competition_frame = create_data_structures(items_id_list, user_id_list, args.dim, args.u_std, args.x_std,
                                                                  u_star, args.num_in_list)
    batch_size = args.num_users
    f_star = FStartSynthetic(args.dim)
    model = OneBitUserPreferenceEncoder(user_id_list, args.dim, alpha=args.alpha, seed=seed)
    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate, betas=(0.9, 0.999))
    loss = Loss(args.dim, top_k=k, sig_tau=args.tau_sig, tau_ndcg=args.tau_ndcg, tau_div=args.tau_div)
    trainer = NdcgDivTrainer(model, loss, optimizer, device, f_star, lambda_=args.lambda_, batch_size=batch_size)
    if args.lambda_ > 0:
        iter_data_train = IterableStrategicYelpData(model, user_id_list, pref_df, f_star, device,
                                                    batch_size,
                                                    items_df, competition_frame=competition_frame)
    else:
        iter_data_train = IterableYelpData(model, user_id_list, pref_df, f_star, device, batch_size,
                                           competition_frame=competition_frame)
    ndcg_score_, div_score_, _ = trainer.fit(iter_data_train, args.num_epoch, print_every=1)
    iter_data_strategic_test = IterableStrategicYelpData(model, user_id_list, pref_df, f_star, device,
                                                         batch_size,
                                                         items_df, competition_frame=competition_frame)
    ndcg_after_before_strategic, div_after_strategic, _, _ = trainer.predict(iter_data_strategic_test)
    return ndcg_after_before_strategic, div_after_strategic


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-a", dest='alpha', type=float, required=False, default=0)
    parser.add_argument("--list", dest='num_in_list', type=int, required=False, default=10)
    parser.add_argument("-nu", dest='num_users', type=int, required=False, default=200)
    parser.add_argument("-ni", dest='num_items', type=int, required=False, default=50)
    parser.add_argument("--lambda", dest='lambda_', type=float, required=False, default=0)
    parser.add_argument("--ustd", dest='u_std', type=float, required=False, default=0.1)
    parser.add_argument("--xstd", dest='x_std', type=float, required=False, default=1)
    parser.add_argument("-ne", dest='num_epoch', type=int, required=False, default=500)
    parser.add_argument("--tau_sig", dest="tau_sig", type=float, required=False, default=2)
    parser.add_argument("--tau_div", dest="tau_div", type=float, required=False, default=5)
    parser.add_argument("--tau_ndcg", dest="tau_ndcg", type=float, required=False, default=5)
    parser.add_argument("--lr", dest="learn_rate", type=float, required=False, default=0.01)
    parser.add_argument("--dim", dest='dim', type=int, required=False, default=2)
    parser.add_argument("-k", dest='k', type=int, required=False, default=None)
    parser.add_argument("--iter", dest='num_iters', type=int, required=False, default=1)
    return parser.parse_args()


def run_exp():
    args = get_args()
    k = args.k if args.k is not None else args.num_in_list
    u_std = args.u_std
    print(f'current users std {u_std}', flush=True)
    div_list = []
    ndcg_list = []
    for iteration in range(args.num_iters):
        random.seed(iteration)
        np.random.seed(iteration)
        ndcg, div = run_exp_on_random_graph(args, k, seed=iteration)
        div_list.append(div)
        ndcg_list.append(ndcg)
        print(f'iter results: ndcg: {ndcg} div:{div}', flush=True)
    div_array = np.array(div_list)
    div_mean, div_std = div_array.mean(), div_array.std()
    ndcg_array = np.array(ndcg_list)
    ndcg_mean, ndcg_std = ndcg_array.mean(), ndcg_array.std()
    print(f'\u03B1: {args.alpha} num in list: {args.num_in_list} \u03BB: {args.lambda_}')
    print(f'users std: {u_std} ndcg mean: {ndcg_mean} ndcg std: {ndcg_std} div mean: {div_mean} div std: {div_std} k: {k}')


if __name__ == '__main__':
    run_exp()
