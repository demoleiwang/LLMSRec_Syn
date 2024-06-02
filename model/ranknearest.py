from .rank import Rank
import numpy as np 
from tqdm import tqdm
from recbole.utils import set_color
import random
import torch
import os 
import json

class RankNearest(Rank):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)


    def demo_construction(self, valid_data, selected_uids, show_progress=True):

        train_iter_data = (
            tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Train   ", "pink"),
            )
            if show_progress
            else valid_data
        )
        training_user_seqs = []
        training_user_pos_items = []
        training_user_ids = []
        train_prs = []
        count = 0
        for batch_idx, batched_data in enumerate(train_iter_data):
            interaction, history_index, positive_u, positive_i = batched_data
            user_ids = interaction['user_id']
            training_user_ids.extend(user_ids.tolist())
            pos_items = interaction[self.POS_ITEM_ID]
            item_seq_lens = interaction[self.ITEM_SEQ_LEN]
            item_seqs = interaction[self.ITEM_SEQ]
            # training_user_seqs.extend(item_seqs)
            training_user_pos_items.extend(pos_items)
            for train_i in range(len(item_seqs)):

                if interaction['user_id'][train_i].item() in selected_uids:
                    train_pr = selected_uids.index(interaction['user_id'][train_i].item())
                    train_prs.append((count, train_pr))

                train_real_len = min(self.config['max_his_len'], item_seq_lens[train_i].item())
                # if train_real_len<10:
                #     continue
                training_user_seqs.append(item_seqs[train_i].detach().cpu().tolist()[:train_real_len])
                count+=1
            # break
        train_prs.sort(key=lambda t: t[1])
        train_prsx = [_[0] for _ in train_prs]

        demo_elems = []
        num_demo = self.config['num_demo_out']  # int(self.version.split('_')[-1])
        user_matrix_aug_sim = self.get_similarity_matrix()

        row_array = np.array(list(range(len(user_matrix_aug_sim)))[:self.config["num_data"]])
        selected_uids_array = np.array(train_prsx)
        user_matrix_aug_sim[row_array, selected_uids_array] = 0.

        item_text = self.item_text
        for test_ui in range(len(selected_uids)):
            sim_per_xx_ids = np.argsort(-user_matrix_aug_sim[test_ui])
            sample_demo_ids = sim_per_xx_ids[0:num_demo]
            demo_elem = []
            # print (test_ui, '-', [item_text[simi_seq_id] for simi_seq_id in selected_user_seqs[test_ui]][-10:])
            for sim_per_xx_idx in sample_demo_ids[:]:
                simi_seq = training_user_seqs[sim_per_xx_idx]
                simi_seq_pos_item = training_user_pos_items[sim_per_xx_idx]
                simi_seq_text = [item_text[simi_seq_id] for simi_seq_id in simi_seq]
                # print ('=',simi_seq_text[-10:], '\n')
                simi_seq_pos_item_text = item_text[simi_seq_pos_item]

                possible_candiates = item_text[1:simi_seq_pos_item] + item_text[simi_seq_pos_item+1:]
                random.seed(self.config['seed'])
                possible_candiates = random.sample(possible_candiates, 19)

                candidates_4_rank_text = [simi_seq_pos_item_text] + possible_candiates[:]
                candidates_4_rank_gt_text = candidates_4_rank_text[:]
                random.seed(self.config['seed'])
                random.shuffle(candidates_4_rank_text)
                
                demo_elem.append([simi_seq_text, candidates_4_rank_text, candidates_4_rank_gt_text])
                # demo_elem.append([simi_seq, simi_seq_pos_item])
            demo_elems.append(demo_elem)

        return demo_elems