from .rank import Rank
import numpy as np 
from tqdm import tqdm
from recbole.utils import set_color
import random
import torch
import os 
import json

class RankZero(Rank):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)


    def prompt_func(self, dataset_name, user_his_text, candidate_text_order, recall_budget):
        if dataset_name in ['ml-1m', 'ml-01m']:
            prompt_zs = f"The User's Movie Profile:\n- Watched Movies: {user_his_text}\n\nThe User's Potential Matches:\n- Candidate Movies: {candidate_text_order}\n\nBased on the user's watched movies, please rank the candidate movies \\nthat align closely with the user's preferences.\n\nPresent your response in the format below:\n1. [Top Recommendation]\n2. [2nd Recommendation]\n...\n{recall_budget}. [{recall_budget}th Recommendation]\n\n"
            
        elif dataset_name == "Games":
            prompt_zs = f"User's Game Product Purchase History:\n- Previously Purchased Game Products (in order of purchase): {user_his_text}\n\nPotential Game Product Recommendations:\n- List of Candidate Game Products for Consideration: {candidate_text_order}\n\nConsidering the user's game product purchase history, please rank the game products from the list of candidate game products to align with the user's preferences. Ensure that:\n- You ONLY rank the given candidate game products.\n- You DO NOT generate game products outside of the listed candidate game products.\n\nFormat your recommendations as follows:\n1. [Top Recommendation]\n2. [2nd Recommendation]\n...\n{recall_budget}. [{recall_budget}th Recommendation]\n"

        elif dataset_name == "lastfm":
            prompt_zs = f"User's Music Listening History:\n- Sequentially listed artists recently listened to by the user: {user_his_text}\n\nList of Potential Recommended Artists:\n- Artists under consideration for recommendation: {candidate_text_order}\n\nBased on the user's music listening history, evaluate and rank the candidate music artists to align with the user's preferences. Ensure that:\n- Only artists from the candidate list are included in the recommendations.\n- No new artists outside the candidate list are introduced.\n\nRecommendations should be formatted as follows:\n1. [Most Recommended Artist]\n2. [Second Most Recommended Artist]\n...\n{recall_budget}. [{recall_budget}th Recommended Artist]\n\n"

        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt_zs

    
    def predict_on_subsets(self, interaction, idxs, valid_data, selected_uids):

        origin_batch_size = idxs.shape[0]
        if self.boots:
            """ 
            bootstrapping is adopted to alleviate position bias
            `fix_enc` is invalid in this case"""
            idxs = np.tile(idxs, [self.boots, 1])
            np.random.shuffle(idxs.T)
        batch_size = idxs.shape[0]
        pos_items = interaction[self.POS_ITEM_ID]

        result_dict = {}
        prompt_list = []
        usef_list = []
        for i in tqdm(range(batch_size)):
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(interaction,
                                                                                                        idxs, i)
            input_ = self.prompt_func(self.dataset_name, user_his_text, candidate_text_order,
                                                self.recall_budget)

            prompt_list.append([{'role': 'user', 'content': input_}])

            result_dict[i] = {"index": i, "input": input_, 'user_his_text':  user_his_text}

        openai_responses = []
        for i in range(0, batch_size, self.api_batch):
            end_i = min(i + self.api_batch, 1000)
            self.logger.info(f'Here are index from {i} to {end_i}')

            prompt_list_batch = prompt_list[i:end_i]
            openai_response_batch = self.dispatch_openai_api_requests(prompt_list_batch, batch_size)
            openai_responses.extend(openai_response_batch)

        scores = torch.full((idxs.shape[0], self.n_items), -10000.)
        for i, openai_response in enumerate(tqdm(openai_responses)):
            user_his_text, candidate_text, candidate_text_order, candidate_idx = self.get_batch_inputs(interaction,
                                                                                                        idxs, i)

            response = openai_response.choices[0].message.content  # openai_response['choices'][0]['message']['content']
            response_list = response.split('\n')

            result_dict[i]['output'] = response

            rec_item_idx_list = self.parsing_output_text(scores, i, response_list, idxs, candidate_text)

            if int(pos_items[i % origin_batch_size]) in candidate_idx:
                target_text = candidate_text[candidate_idx.index(int(pos_items[i % origin_batch_size]))]
                try:
                    ground_truth_pr = rec_item_idx_list.index(target_text)
                    # self.logger.info(f'Ground-truth [{target_text}]: Ranks {ground_truth_pr}')
                    # self.logger.info(f'Ranking List [{rec_item_idx_list}]')
                except:
                    # self.logger.info(f'Fail to find ground-truth items.')
                    print(target_text)
                    print(rec_item_idx_list)
                    ground_truth_pr = -1

            result_dict[i]['prediction'] = rec_item_idx_list
            # result_dict[i]['prediction'] = rec_item_idx_list
            result_dict[i]['candidate'] = candidate_text
            result_dict[i]['target_text'] = target_text
            result_dict[i]['Ranking Position'] = ground_truth_pr
            result_dict[i]['Overlap_list'] = list(set(rec_item_idx_list).intersection(candidate_text))
            result_dict[i]['Overlap_len'] = len(set(rec_item_idx_list).intersection(candidate_text))

        prompt_merge_path = os.path.join(self.data_path, "output_files",
                                        f"final11_{self.api_model_name}_{self.dataset_name}_{self.config.model}_output_{self.config['seed']}.json")
        with open(prompt_merge_path, mode='w') as f:
            json.dump(result_dict, f, indent=2)
        self.logger.info(f"SAVE DONE! Path: {prompt_merge_path}")

        if self.boots:
            scores = scores.view(self.boots, -1, scores.size(-1))
            scores = scores.sum(0)
        return scores 