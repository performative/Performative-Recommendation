from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--ndcg_target", dest="ndcg_target", type=float, required=False, default=None)
    parser.add_argument("--ndcg_target_list", dest='ndcg_target_list', nargs='*', type=float, required=False, default=[])
    parser.add_argument("-ne", dest='num_epoch', type=int, required=False, default=100)
    parser.add_argument("--strategic_model", dest='strategic_model', action='store_true', required=False, default=False)
    parser.add_argument("--test", dest="test", action='store_true', required=False, default=False)
    parser.add_argument("--calc_post_items", dest='calc_post_items', action='store_true', required=False, default=False)
    parser.add_argument("--batch_size", dest="batch_size", type=int, required=False, default=64)
    parser.add_argument("--ndcg_div_lr", dest="learn_rate", type=float, required=False, default=0.1)
    parser.add_argument("-k", dest="k", type=int, required=False, default=10)
    parser.add_argument("--tau_sig", dest="tau_sig", type=float, required=False, default=2)
    parser.add_argument("--tau_div", dest="tau_div", type=float, required=False, default=1)
    parser.add_argument("--tau_ndcg", dest="tau_ndcg", type=float, required=False, default=0.1)
    parser.add_argument("-a", dest="alpha", type=float, required=False, default=0.1)
    parser.add_argument("--start_time", dest='start_time', type=int, required=False, default=0)
    parser.add_argument("--end_time", dest='end_time', type=int, required=False, default=1)
    parser.add_argument("--random", dest='random', action='store_true', required=False, default=False)



    return parser.parse_args()