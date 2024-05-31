import os
import numpy as np
from tqdm import tqdm
import torch
from recbole.trainer import Trainer
from recbole.utils import EvaluatorType, set_color
from recbole.data.interaction import Interaction

from sklearn.metrics.pairwise import cosine_similarity
import random

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.spatial.distance import cdist

import json


class SelectedUserTrainer(Trainer):
    def __init__(self, config, model, dataset):
        super().__init__(config, model)
        self.selected_user_suffix = config['selected_user_suffix']  # candidate generation model, by default, random
        self.recall_budget = config['recall_budget']                # size of candidate Sets, by default, 20
        self.fix_pos = config['fix_pos']                            # whether fix the position of ground-truth items in the candidate set, by default, -1
        self.selected_uids, self.sampled_items = self.load_selected_users(config, dataset)

        self.USER_ID = config["USER_ID_FIELD"]
        self.ITEM_ID = config["ITEM_ID_FIELD"]
        self.ITEM_SEQ = self.ITEM_ID + config["LIST_SUFFIX"]
        self.ITEM_SEQ_LEN = config["ITEM_LIST_LENGTH_FIELD"]
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config["NEG_PREFIX"] + self.ITEM_ID

    def load_selected_users(self, config, dataset):
        selected_users = []
        sampled_items = []
        selected_user_file = os.path.join(config['data_path'], f'{config["dataset"]}.{self.selected_user_suffix}')
        user_token2id = dataset.field2token_id['user_id']
        item_token2id = dataset.field2token_id['item_id']
        count = 0
        with open(selected_user_file, 'r', encoding='utf-8') as file:
            for line in file:
                uid, iid_list = line.strip().split('\t')
                selected_users.append(uid)
                sampled_items.append([item_token2id[_] if (_ in item_token2id) else 0 for _ in iid_list.split(' ')])
                count+=1
                if count >=self.config['num_data']:
                    break
        selected_uids = list([user_token2id[_] for _ in selected_users])
        return selected_uids, sampled_items

    @torch.no_grad()
    def evaluate(
        self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        self.model.eval()
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )
        unsorted_selected_interactions = []
        unsorted_selected_pos_i = []
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, history_index, positive_u, positive_i = batched_data
            for i in range(len(interaction)):
                if interaction['user_id'][i].item() in self.selected_uids:
                    pr = self.selected_uids.index(interaction['user_id'][i].item())
                    unsorted_selected_interactions.append((interaction[i], pr))
                    unsorted_selected_pos_i.append((positive_i[i], pr))
        unsorted_selected_interactions.sort(key=lambda t: t[1])
        unsorted_selected_pos_i.sort(key=lambda t: t[1])
        selected_interactions = [_[0] for _ in unsorted_selected_interactions]
        selected_pos_i = [_[0] for _ in unsorted_selected_pos_i]
        new_inter = {
            col: torch.stack([inter[col] for inter in selected_interactions]) for col in selected_interactions[0].columns
        }
        selected_interactions = Interaction(new_inter)
        selected_pos_i = torch.stack(selected_pos_i)
        selected_pos_u = torch.arange(selected_pos_i.shape[0])

        if self.config['has_gt']: # should be true here.
            self.logger.info('Has ground truth.')
            idxs = torch.LongTensor(self.sampled_items)
            for i in range(idxs.shape[0]):
                if selected_pos_i[i] in idxs[i]:
                    pr = idxs[i].numpy().tolist().index(selected_pos_i[i].item())
                    idxs[i][pr:-1] = torch.clone(idxs[i][pr+1:])

            idxs = idxs[:,:self.recall_budget - 1]
            if self.fix_pos == -1 or self.fix_pos == self.recall_budget - 1:
                idxs = torch.cat([idxs, selected_pos_i.unsqueeze(-1)], dim=-1).numpy()
            elif self.fix_pos == 0:
                idxs = torch.cat([selected_pos_i.unsqueeze(-1), idxs], dim=-1).numpy()
            else:
                idxs_a, idxs_b = torch.split(idxs, (self.fix_pos, self.recall_budget - 1 - self.fix_pos), dim=-1)
                idxs = torch.cat([idxs_a, selected_pos_i.unsqueeze(-1), idxs_b], dim=-1).numpy()
        else:
            self.logger.info('Does not have ground truth.')
            idxs = torch.LongTensor(self.sampled_items)
            idxs = idxs[:,:self.recall_budget]
            idxs = idxs.numpy()

        if self.fix_pos == -1: # should be -1
            self.logger.info('Shuffle ground truth')
            for i in range(idxs.shape[0]):
                np.random.shuffle(idxs[i])
       
        scores = self.model.predict_on_subsets(selected_interactions.to(self.device), idxs)
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        self.eval_collector.eval_batch_collect(
            scores, selected_interactions, selected_pos_u, selected_pos_i
        )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result

    @torch.no_grad()
    def evaluatex(
            self, eval_data, load_best_model=True, model_file=None, show_progress=False
    ):
        self.model.eval()
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = eval_data._dataset.item_num

        iter_data = (
            tqdm(
                eval_data,
                total=len(eval_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
            if show_progress
            else eval_data
        )
        eval_func = self._full_sort_batch_eval
        unsorted_selected_interactions = []
        unsorted_selected_pos_i = []
        for batch_idx, batched_data in enumerate(iter_data):
            interaction, history_index, positive_u, positive_i = batched_data
            for i in range(len(interaction)):
                if interaction['user_id'][i].item() in self.selected_uids:
                    pr = self.selected_uids.index(interaction['user_id'][i].item())
                    unsorted_selected_interactions.append((interaction[i], pr))
                    unsorted_selected_pos_i.append((positive_i[i], pr))
        unsorted_selected_interactions.sort(key=lambda t: t[1])
        unsorted_selected_pos_i.sort(key=lambda t: t[1])
        selected_interactions = [_[0] for _ in unsorted_selected_interactions]
        selected_pos_i = [_[0] for _ in unsorted_selected_pos_i]
        new_inter = {
            col: torch.stack([inter[col] for inter in selected_interactions]) for col in
            selected_interactions[0].columns
        }
        selected_interactions = Interaction(new_inter)
        selected_pos_i = torch.stack(selected_pos_i)
        selected_pos_u = torch.arange(selected_pos_i.shape[0])

        if self.config['has_gt']:
            self.logger.info('Has ground truth.')
            idxs = torch.LongTensor(self.sampled_items)
            for i in range(idxs.shape[0]):
                if selected_pos_i[i] in idxs[i]:
                    pr = idxs[i].numpy().tolist().index(selected_pos_i[i].item())
                    idxs[i][pr:-1] = torch.clone(idxs[i][pr + 1:])

            idxs = idxs[:, :self.recall_budget - 1]
            if self.fix_pos == -1 or self.fix_pos == self.recall_budget - 1:
                idxs = torch.cat([idxs, selected_pos_i.unsqueeze(-1)], dim=-1).numpy()
            elif self.fix_pos == 0:
                idxs = torch.cat([selected_pos_i.unsqueeze(-1), idxs], dim=-1).numpy()
            else:
                idxs_a, idxs_b = torch.split(idxs, (self.fix_pos, self.recall_budget - 1 - self.fix_pos), dim=-1)
                idxs = torch.cat([idxs_a, selected_pos_i.unsqueeze(-1), idxs_b], dim=-1).numpy()
        else:
            self.logger.info('Does not have ground truth.')
            idxs = torch.LongTensor(self.sampled_items)
            idxs = idxs[:, :self.recall_budget]
            idxs = idxs.numpy()

        if self.fix_pos == -1:
            self.logger.info('Shuffle ground truth')
            for i in range(idxs.shape[0]):
                np.random.shuffle(idxs[i])
        
        scores = self.model.full_sort_predict(selected_interactions.to(self.device))
        scores = scores.view(-1, self.tot_item_num)
        scores[:, 0] = -np.inf
        self.eval_collector.eval_batch_collect(
            scores, selected_interactions, selected_pos_u, selected_pos_i
        )
        self.eval_collector.model_collect(self.model)
        struct = self.eval_collector.get_data_struct()
        result = self.evaluator.evaluate(struct)
        self.wandblogger.log_eval_metrics(result, head="eval")
        return result

    def multi_hot_similarity(self, test_list_x, train_list_x):
        item_text = self.model.item_text
        file_ = os.path.join(self.config['data_path'], "user_matrix_aug_sim.npy")
        print (file_, os.path.exists(file_))


        test_user_list = []
        for i, test_seq in enumerate(test_list_x):
            item_hot_list = [0. for ii in range(len(item_text))]
            for item_pos in test_seq:
                item_hot_list[item_pos] = 1.
            test_user_list.append(item_hot_list)
        test_user_matrix = np.array(test_user_list)
        normalized_test_user_matrix = test_user_matrix

        train_user_list = []
        for i, train_seq in enumerate(train_list_x):
            item_hot_list = [0. for ii in range(len(item_text))]
            for item_pos in train_seq:
                item_hot_list[item_pos] = 1.
            train_user_list.append(item_hot_list)
        train_user_matrix = np.array(train_user_list)
        normalized_train_user_matrix = train_user_matrix  # / np.sum(train_user_matrix, axis=1)[:, np.newaxis]

        if os.path.exists(file_):
            user_matrix_aug_sim = np.load(file_)
            print (f"loading {file_}")
        else:
             #/ np.sum(test_user_matrix, axis=1)[:, np.newaxis]

            print ("cosine_similarity...")
            user_matrix_aug_sim = cosine_similarity(normalized_test_user_matrix, normalized_train_user_matrix)

            np.save(file_, user_matrix_aug_sim)
            print ('saved')
        
        return user_matrix_aug_sim, normalized_test_user_matrix, normalized_train_user_matrix

    @torch.no_grad()
    def pop_method(
            self, train_data, show_progress=False
    ):
        self.model.eval()
        item_text = self.model.item_text
        if self.config["eval_type"] == EvaluatorType.RANKING:
            self.tot_item_num = train_data._dataset.item_num

        train_iter_data = (
            tqdm(
                train_data,
                total=len(train_data),
                ncols=100,
                desc=set_color(f"Train   ", "pink"),
            )
            if show_progress
            else train_data
        )
        training_user_seqs = []
        training_user_pos_items = []
        pop_item_feq = {}
        total_items = 0
        for batch_idx, interaction in enumerate(train_iter_data):
            # interaction, history_index, positive_u, positive_i = batched_data
            pos_items = interaction[self.POS_ITEM_ID]
            item_seq_lens = interaction[self.ITEM_SEQ_LEN]
            item_seqs = interaction[self.ITEM_SEQ]
            # training_user_seqs.extend(item_seqs)
            training_user_pos_items.extend(pos_items)
            for train_i in range(len(item_seqs)):
                # train_real_len = min(self.config['max_his_len'], item_seq_lens[train_i].item())
                # if train_real_len<10:
                #     continue
                # training_user_seqs.append(item_seqs[train_i].detach().cpu().tolist()[:train_real_len])
                item_seq_text = [item_text[item_x] for item_x in item_seqs[train_i].detach().cpu().tolist()[:50]]
                for item_string in item_seq_text:
                    if item_string in pop_item_feq:
                        pop_item_feq[item_string] +=1
                    else:
                        pop_item_feq[item_string] = 1
                    total_items+=1
                training_user_seqs.append(item_seq_text)
                # print (item_seq_text)
        pop_item_feq_sorted_list = sorted(list(pop_item_feq.items()), key=lambda x:x[1], reverse=True)
        print (pop_item_feq_sorted_list[:100])
        pop_list_file = os.path.join(self.config['data_path'], f'{self.config["dataset"]}.popular_list.json')
        with open(pop_list_file, 'w') as f:
            json.dump(pop_item_feq_sorted_list, f)
        # pop_item_feq_sorted_list
        # assert 1==0
        print ("pop_list_file save done!")

