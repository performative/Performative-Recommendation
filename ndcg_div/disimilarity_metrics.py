import torch


def dissimilarity_l2(query_batch, device) -> torch.tensor:
    query_batch = query_batch.to(device)
    similarity_batch = torch.norm(query_batch.unsqueeze(1) - query_batch.unsqueeze(-2), dim=-1)
    return similarity_batch


def cosine_similarity(query_batch, device) -> torch.tensor:
    '''

    :param query_batch: batch from the shape: (num_queries, max_num_items, num_features)
    :return: tensor from the shape: (num_queries, max_num_items, max_num_items) which each cell [q, i , j] is the cosine
    similarity between item i and j in query q
    '''
    query_batch = query_batch.to(device)
    norm_items_vector = torch.norm(query_batch, 2, dim=2)
    norm_items_matrix = norm_items_vector.unsqueeze(2) @ norm_items_vector.unsqueeze(1)
    similarity_batch = query_batch @ query_batch.transpose(1, 2) * (1 / norm_items_matrix)
    return similarity_batch


def dissimilarity_cosine(query_batch, device) -> torch.tensor:
    '''

        :param query_batch: batch from the shape: (num_queries, max_num_items, num_features)
        :return: tensor from the shape: (num_queries, max_num_items, max_num_items) which each cell [q, i , j] is the cosine
        dissimilarity between item i and j in query q number between 0 to 1.
        '''
    similarity_batch = cosine_similarity(query_batch, device)
    dissimilarity_batch = (-1 * similarity_batch + 1) / 2
    return dissimilarity_batch
