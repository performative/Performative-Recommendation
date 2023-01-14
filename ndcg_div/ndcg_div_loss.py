import torch
from ndcg_div.disimilarity_metrics import dissimilarity_cosine as dissimilarity_metric


class Loss(torch.nn.Module):
    def __init__(self, first_user_feature: int, discount_function: str ='log', top_k: int = 10, sig_tau: float = 1,
                 tau_ndcg: float = 0.1, tau_div: float = 0.1):
        '''

        :param first_user_feature: The first index of the user features. Each feature vector is stack of item feature
        and user features
        :param discount_function: The discount function in the NDCG calculation. can be 'log' or 'identity'
        :param top_k: The number of items to include in NDCG and diversity calculation
        :param sig_tau: The temperature parameter for sigmoid function in the diversity calculation
        :param tau_ndcg: The temperature parameter for the soft permutation matrix in the NDCG calculation
        :param tau_div: The temperature parameter for the soft permutation matrix in the diversity calculation
        '''
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.first_user_feature = first_user_feature
        if discount_function == 'identity':
            self.discount_function = lambda relevance: torch.arange(1, relevance.shape[1] + 1).float().to(self.device)
        elif discount_function == 'log':
            self.discount_function = lambda relevance: torch.log2(torch.arange(2, relevance.shape[1] + 2).float().to(self.device))
        else:
            print('illegal discount function exit from the program')
            exit(-2)
        self.k = top_k
        self.sig_tau = sig_tau
        self.tau_ndcg = tau_ndcg
        self.tau_div = tau_div

    @staticmethod
    def sinkhorn_scaling(mat: torch.tensor, tol: float = 1e-6, max_iter: int = 5, default_eps: float = 1e-10) -> torch.tensor:
        """
        Sinkhorn scaling procedure.
        :param mat: a tensor of square matrices of shape N x M x M, where N is batch size

        :param tol: Sinkhorn scaling tolerance
        :param max_iter: maximum number of iterations of the Sinkhorn scaling
        :return: a tensor of (approximately) doubly stochastic matrices
        """
        for _ in range(max_iter):
            mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=default_eps)
            mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=default_eps)
            if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
                break
        return mat

    @staticmethod
    def calc_p_sort(model_score: torch.tensor, device: torch.cuda.device, tau: float = 0.1,
                    use_sinkhorn_scaling: bool = True) -> torch.tensor:
        '''

        :param model_score: The score that each item got by the model.
        :param device: device for computation
        :param tau: hyper param for the smoothness of p
        :return: a permutation matrix p
        '''

        model_score = model_score.unsqueeze(-2).float()
        # batch.shape  = (batch, 1, n)
        minus_1_tensor = (torch.ones_like(model_score.transpose(-2, -1)) * (-1)).to(device)

        a_s = (minus_1_tensor @ model_score + model_score.transpose(-2, -1)).abs()
        # (n + 1 - 2i)s where we write it as vector.
        tmp = (model_score.shape[-1] + 1 - 2 * torch.arange(1, model_score.shape[-1] + 1, device=device)).repeat(
            *model_score.shape[0:-2], 1).unsqueeze(-1).float() @ model_score
        p_sort = torch.softmax((tmp - a_s.sum(dim=-1).unsqueeze(-2)) / tau, dim=-1)
        if use_sinkhorn_scaling:
            p_sort = Loss.sinkhorn_scaling(p_sort)
        return p_sort

    def calc_dcg(self, relevance: torch.tensor) -> torch.tensor:
        '''

        :param relevance: The of the items. should be in shape of (num_users, num_items).
        :return: The dcg of the list. The gain function is just the relevance and the discounted function the determined
        by the location of the item in the list.
        '''
        if self.k is not None:
            relevance = relevance[:, :self.k]
        dcg = ((2**relevance - 1) / self.discount_function(relevance)).sum(axis=1)
        return dcg

    def calc_ndcg_loss(self, p_sort_matrix: torch.tensor, model_score: torch.tensor, relevance: torch.tensor):
        '''

        :param p_sort_matrix: permutation matrices. Each matrix is for a different query. should be from the shape: (num_query, num item, num item)
        :param model_score: he score that each item got by the model. should be from the shape: (num_query, num item)
        :param relevance: The of the items. should be in shape of (num_query, num_items).
        :return:
        '''
        approx_relevance_sorted_by_f = p_sort_matrix @ relevance.unsqueeze(2).float().to(self.device)
        # relevance_sorted_by_f is shaped: bacth_size, num of item, 1

        approx_dcg_score = self.calc_dcg(approx_relevance_sorted_by_f.float().squeeze(2))
        relevance_sorted_by_model = relevance.gather(dim=1, index=model_score.sort(dim=1, descending=True)[1])

        dcg_score = self.calc_dcg(relevance_sorted_by_model)
        best_dcg_score = self.calc_dcg(relevance.sort(dim=1, descending=True)[0])
        ndcg_score = dcg_score / best_dcg_score
        return approx_dcg_score / best_dcg_score, ndcg_score

    @staticmethod
    def zero_below_diagonal(matrix_batch):
        for i in range(matrix_batch.shape[1]):
            matrix_batch[:, i, :i + 1] = 0
        return matrix_batch

    @staticmethod
    def create_rank_matrix(rank_vector: torch.tensor) -> torch.tensor:
        rank_matrix = rank_vector @ rank_vector.transpose(1, 2)
        rank_matrix = Loss.zero_below_diagonal(rank_matrix)
        return rank_matrix

    @staticmethod
    def get_approx_rank_vector(p_sort_matrix: torch.tensor, device: torch.cuda.device) -> torch.tensor:
        num_item = p_sort_matrix.shape[-2]
        approx_rank_vector = (p_sort_matrix * torch.arange(start=num_item, end=0, step=-1, device=device).unsqueeze(1)).sum(dim=-2)
        return approx_rank_vector

    def calc_diversity_reg(self, batch: torch.tensor, model_score: torch.tensor, p_sort_matrix: torch.tensor) -> \
            (torch.tensor, torch.tensor):
        batch = batch[:, :, :self.first_user_feature]
        num_item = batch.shape[1]
        # rank means that the best item has the highest rank. that means if we have 5 items and the best item in
        # position 2 it will get rank 5
        approx_rank_vector = self.get_approx_rank_vector(p_sort_matrix, self.device)
        if self.k is not None:
            approx_rank_vector = torch.sigmoid(self.sig_tau * (approx_rank_vector - (num_item - self.k)))

        approx_rank_matrix = Loss.create_rank_matrix(approx_rank_vector.unsqueeze(2))
        real_rank_tensor_list = list()
        for i in range(model_score.shape[0]):
            new_tensor = torch.empty(model_score.shape[1], device=self.device)
            new_tensor[model_score[i].argsort()] = torch.arange(start=1, end=model_score[0].shape[0] + 1, device=self.device).float()
            real_rank_tensor_list.append(new_tensor)
        real_rank_by_model_vector = torch.stack(real_rank_tensor_list, dim=0)
        real_rank_by_model_vector = real_rank_by_model_vector.unsqueeze(2)

        if self.k is not None:
            real_rank_by_model_vector = (real_rank_by_model_vector - (num_item - self.k)).clip(min=0).clip(max=1)
        real_rank_by_model_matrix = Loss.create_rank_matrix(real_rank_by_model_vector)
        disimilarity_batch = dissimilarity_metric(batch, self.device)
        clean_disimilarity_batch = Loss.zero_below_diagonal(disimilarity_batch)
        approx_diversity = (approx_rank_matrix * clean_disimilarity_batch).sum(dim=(1, 2)) / (
            approx_rank_matrix.sum(dim=(1, 2)).detach())
        diversity_score = (real_rank_by_model_matrix * clean_disimilarity_batch).sum(dim=(1, 2)) / (
            real_rank_by_model_matrix.sum(dim=(1, 2)))
        return approx_diversity, diversity_score



    def forward(self, batch: torch.tensor, model_score_for_div: torch.tensor, model_score_for_ndcg: torch.tensor,
                relevance: torch.tensor) -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        '''
        :param batch: Should be from the shape: (num of user in batch, num of item, num of items to user)
        :param model_score: Score of each item in each user by the recommendation system. Should be from the shape:
        (num of users in batch, num of item)
        :param relevance: Relevance of each item in each user. This is the ground truth. Should be from the shape:
        (num of users in batch, num of item)
        :return: loss tensor.
        '''
        p_sort_for_ndcg = Loss.calc_p_sort(model_score_for_ndcg, self.device, self.tau_ndcg)
        approx_ndcg, ndcg_score = self.calc_ndcg_loss(p_sort_for_ndcg, model_score_for_ndcg, relevance)
        p_sort_for_div = self.calc_p_sort(model_score_for_div, self.device, self.tau_div)
        approx_diversity, diversity_score = self.calc_diversity_reg(batch, model_score_for_div, p_sort_for_div)
        return approx_ndcg, ndcg_score, approx_diversity, diversity_score



