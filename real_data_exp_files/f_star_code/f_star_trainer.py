import torch
from tqdm import tqdm
import copy
from models import MLP


class FStarTrainer:
    def __init__(self, model: MLP, optimizer: torch.optim, device: torch.cuda.device):
        self.loss_func = torch.nn.BCELoss()
        self.optimizer = optimizer
        self.device = device
        self.model = model.to(self.device)

    def for_each_batch(self, batch_x: torch.tensor, batch_y: torch.tensor, pred_rate_list: list, true_y_list: list,
                       loss_list: list) -> torch.tensor:
        if type(batch_x) is tuple:
            rate_pred = self.model((batch_x[0], batch_x[1].to(self.device)))
        else:
            rate_pred = torch.nn.Sigmoid()(self.model(batch_x.to(self.device))).squeeze(1)
        pred_rate_list.extend((rate_pred > 0.5).long().tolist())
        true_y_list.extend(batch_y.tolist())
        loss = self.loss_func(rate_pred, batch_y.to(self.device))
        loss_list.append(loss.item())
        return loss

    def for_each_batch_train(self, batch_x: torch.tensor, batch_y: torch.tensor, pred_rate_list: list, true_y_list: list
                             , loss_list: list) -> None:
        self.optimizer.zero_grad()
        loss = self.for_each_batch(batch_x, batch_y, pred_rate_list, true_y_list, loss_list)
        loss.backward()
        self.optimizer.step()

    def for_each_epoch_train(self, train_data_gen, pred_rate_list, true_y_list, train_loss_list, should_print):
        self.model.train()
        for batch_x, batch_y in train_data_gen:
            self.for_each_batch_train(batch_x, batch_y, pred_rate_list, true_y_list, train_loss_list)
        if should_print:
            acc = (torch.tensor(pred_rate_list) == torch.tensor(true_y_list)).sum() / len(true_y_list)
            print(f'train acc: {acc}')

    def for_each_epoch_eval(self, val_data_gen, pred_rate_list, true_y_list, val_loss_list, should_print):
        self.model.eval()
        for batch_x, batch_y in val_data_gen:
            self.for_each_batch(batch_x, batch_y, pred_rate_list, true_y_list, val_loss_list)
        acc = (torch.tensor((pred_rate_list)) == torch.tensor(true_y_list)).sum() / len(true_y_list)
        if should_print:
            print(f'val acc: {acc}')
        return acc

    def for_each_epoch(self, batch_gen_creator_func, epoch, print_every, num_epoch, best_res, best_epoch, model_best_state, is_training):
        pred_rate_list = list()
        true_y_list = list()
        loss_list = list()
        data_gen = batch_gen_creator_func()
        should_print = epoch % print_every == 0 or epoch == num_epoch - 1
        if is_training:
            self.for_each_epoch_train(data_gen, pred_rate_list, true_y_list, loss_list, should_print)
        else:
            val_acc = self.for_each_epoch_eval(data_gen, pred_rate_list, true_y_list, loss_list, should_print)
            if best_res < val_acc:
                best_res = val_acc
                best_epoch = epoch
                model_best_state = copy.deepcopy(self.model.state_dict())
        return best_res, best_epoch, model_best_state

    def fit(self, train_data_gen_creator, val_data_gen_creator, num_epoch, print_every=1):
        best_res = 0
        best_epoch = -1
        model_best_state = self.model.state_dict()
        for epoch in tqdm(range(num_epoch)):
            self.for_each_epoch(train_data_gen_creator, epoch, print_every, num_epoch, best_res, best_epoch, model_best_state, is_training=True)
            best_res, best_epoch, model_best_state = self.for_each_epoch(val_data_gen_creator, epoch, print_every, num_epoch, best_res, best_epoch, model_best_state, is_training=False)
        print(f'best acc: {best_res} best epoch: {best_epoch}')
        return model_best_state, best_res, best_epoch

    def eval_on_test(self, test_generator_creator):
        self.model.eval()
        data_gen = test_generator_creator()
        test_acc = self.for_each_epoch_eval(data_gen, [], [], [], should_print=False)
        print(f'f* test acc: {test_acc}', flush=True)
