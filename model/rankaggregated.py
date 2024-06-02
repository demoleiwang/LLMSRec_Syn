from .rank import Rank
import numpy as np 
from tqdm import tqdm
from recbole.utils import set_color
import random

class RankAggregated(Rank):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

    
    def prompt_func(self, dataset_name, user_his_text, candidate_text_order, recall_budget, demo_elems):
        if dataset_name in ['ml-1m', 'ml-01m']:

            demo_prompt = "--------------------------\nDemonstration Example {}:\n\nThe User's Movie Profile:\n- Watched Movies: {}\n\nThe User's Potential Matches:\n- Candidate Movies: {}\n\nBased on the user's watched movies, please rank the candidate movies \\nthat align closely with the user's preferences. \n- You ONLY rank the given Candidate Movies.\n- You DO NOT generate movies from Watched Movies.\n\nPresent your response in the format below:\n1. [Top Recommendation (Candidate Movie)]\n2. [2nd Recommendation (Candidate Movie)]\n...\n20. [20th Recommendation (Candidate Movie)]\n\nAnswer:{}"

            demo_elem_str = ""
            for demo_i, demo_elem in enumerate(demo_elems):
                input_1 = [str(j) + '. ' + demo_elem[0][j] for j in range(len(demo_elem[0]))]
                input_2 = [str(j) + '. ' + demo_elem[1][j] for j in range(len(demo_elem[1]))]

                demo_predict ='\n'.join([str(j+1) + '. ' + demo_elem[2][j] for j in range(len(demo_elem[2]))])
                demo_prompt_str = demo_prompt.format(demo_i, input_1, input_2, demo_predict)
                demo_elem_str += demo_prompt_str

            prompt_icl = f"Demonstration Examples:\n\n{demo_elem_str}\n\n------------------\n\nLearn from the above demonstration examples to solve the following test example\nTest example:\n\nJohn's Movie Profile:\n- Watched Movies: {user_his_text}\n\nJohn's Potential Matches:\n- Candidate Movies: {candidate_text_order}\n\nBased on John's watched movies, please rank the candidate movies \\nthat align closely with John's preferences. \n- You ONLY rank the given Candidate Movies.\n- You DO NOT generate movies from Watched Movies.\n\nPresent your response in the format below:\n1. [Top Recommendation (Candidate Movie)]\n2. [2nd Recommendation (Candidate Movie)]\n...\n{recall_budget}. [{recall_budget}th Recommendation (Candidate Movie)]\n\nAnswer:\n"

        elif dataset_name == "Games":
            demo_prompt = "User's Game Product Purchase History:\n- Previously Purchased Game Products (in order of purchase): {}\n\nPotential Game Product Recommendations:\n- List of Candidate Game Products for Consideration: {}\n\nConsidering the user's past game product purchases, please prioritize the game products from the provided list that best match the user's preferences. Ensure that:\n- You ONLY rank the given candidate game products.\n- You DO NOT generate game products outside of the listed candidate game products.\n\nPresent your response in the format below:\n1. [Top Recommendation]\n2. [2nd Recommendation]\n...\n\n{}\n"

            demo_elem_str = ""
            for demo_i, demo_elem in enumerate(demo_elems):
                # print (demo_elem[2])
                input_1 = [str(j) + '. ' + demo_elem[0][j] for j in range(len(demo_elem[0]))]
                input_2 = [str(j) + '. ' + demo_elem[1][j] for j in range(len(demo_elem[1]))]
                # input_3 = [str(j) + '. ' + demo_elem[2][j] for j in range(len(demo_elem[2]))]
                demo_predict = '\n'.join([str(j + 1) + '. ' + demo_elem[2][j] for j in range(len(demo_elem[2]))])
                demo_prompt_str = demo_prompt.format(input_1, input_2, demo_predict)
                demo_elem_str += demo_prompt_str

            prompt_icl = f"\nExamples:\n{demo_elem_str}\n\n\nLearn from the above examples to solve the following test example\nTest example:\n\nUser's Game Product Purchase History:\n- Previously Purchased Game Products (in order of purchase): {user_his_text}\n\nPotential Game Product Recommendations:\n- List of Candidate Game Products for Consideration: {candidate_text_order}\n\nConsidering the user's past game product purchases, please prioritize the game products from the provided list that best match the user's preferences. Ensure that:\n- You ONLY rank the given candidate game products.\n- You DO NOT generate game products outside of the listed candidate game products.\n\nFormat your recommendations as follows:\n1. [Top Recommendation]\n2. [2nd Recommendation]\n...\n{recall_budget}. [{recall_budget}th Recommendation]\n"

        elif dataset_name == "lastfm":
            demo_prompt = "User's Previously Listened Music Artists:\n- Recently Listened Music Artists (in sequential order): {}\n\nCandidate Music Artists for Recommendation:\n- Candidate Music Artists: {}\n\nGiven the user's listening history, please arrange the candidate music artists in order of relevance to the user's preferences. It is important to:\n- Rank ONLY the music artists listed in the candidates.\n- Avoid introducing any music artists not included in the candidate list.\n\nRecommendations should be formatted as follows:\n1. [Most Recommended Artist]\n2. [Second Most Recommended Artist]\n...\n\n{}"

            demo_elem_str = ""
            for demo_i, demo_elem in enumerate(demo_elems):
                # print (demo_elem[2])
                input_1 = [str(j) + '. ' + demo_elem[0][j] for j in range(len(demo_elem[0]))]
                input_2 = [str(j) + '. ' + demo_elem[1][j] for j in range(len(demo_elem[1]))]
                # input_3 = [str(j) + '. ' + demo_elem[2][j] for j in range(len(demo_elem[2]))]
                demo_predict = '\n'.join([str(j + 1) + '. ' + demo_elem[2][j] for j in range(len(demo_elem[2]))])
                demo_prompt_str = demo_prompt.format(input_1, input_2, demo_predict)
                demo_elem_str += demo_prompt_str

            prompt_icl = f"\nExamples:\n{demo_elem_str}\n\n\nLearn from the above examples to solve the following test example\nTest example:\n\nUser's Previously Listened Music Artists:\n- Recently Listened Music Artists (in sequential order): {user_his_text}\n\nCandidate Music Artists for Recommendation:\n- Candidate Music Artists: {candidate_text_order}\n\nGiven the user's listening history, please arrange the candidate music artists in order of relevance to the user's preferences. It is important to:\n- Rank ONLY the music artists listed in the candidates.\n- Avoid introducing any music artists not included in the candidate list.\n\nRecommendations should be formatted as follows:\n1. [Most Recommended Artist]\n2. [Second Most Recommended Artist]\n...\n{recall_budget}. [{recall_budget}th Recommended Artist]\n"

        else:
            raise NotImplementedError(f'Unknown dataset [{dataset_name}].')
        return prompt_icl


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
        num_demo_out = self.config['num_demo_out']  # int(self.version.split('_')[-1])
        num_demo_int = self.config['num_demo_int']
        user_matrix_aug_sim = self.get_similarity_matrix()

        row_array = np.array(list(range(len(user_matrix_aug_sim)))[:self.config["num_data"]])
        selected_uids_array = np.array(train_prsx)
        user_matrix_aug_sim[row_array, selected_uids_array] = 0.

        item_text = self.item_text
        for test_ui in range(len(selected_uids)):
            sim_per_xx_ids = np.argsort(-user_matrix_aug_sim[test_ui])
            # sample_demo_ids = sim_per_xx_ids[0:num_demo_int*num_demo_out]

            demo_elem = []
            for num_demo_start in range(0, num_demo_int*num_demo_out, num_demo_int):
                num_demo_end = num_demo_start + num_demo_int
                sample_demo_ids = sim_per_xx_ids[num_demo_start:num_demo_end]

                simi_seq_text = []
                for temporal_i in range(int(50 / num_demo_int)):  # int(50/num_demo)
                    for sim_per_xx_idx in sample_demo_ids[:]:
                        simi_seq = training_user_seqs[sim_per_xx_idx]
                        idx_s = len(simi_seq) - 1 - temporal_i
                        try:
                            item_s_text = item_text[simi_seq[idx_s]]
                            simi_seq_text.append(item_s_text)
                        except:
                            pass
                
                simi_seq_pos_item_text = []
                for sim_per_xx_idx in sample_demo_ids[:]:
                    simi_seq_pos_item = training_user_pos_items[sim_per_xx_idx]
                    simi_seq_pos_item_text.append(item_text[simi_seq_pos_item])

                simi_seq_text = simi_seq_text[::-1]
                simi_seq_pos_item_text = simi_seq_pos_item_text[::-1]

                possible_candiates = [item for item in item_text[1:] if item not in simi_seq_pos_item_text]
                random.seed(self.config['seed'])
                possible_candiates = random.sample(possible_candiates, 20-num_demo_int)

                candidates_4_rank_text = simi_seq_pos_item_text + possible_candiates[:]
                candidates_4_rank_gt_text = candidates_4_rank_text[:]
                random.seed(self.config['seed'])
                random.shuffle(candidates_4_rank_text)

                demo_elem.append([simi_seq_text, candidates_4_rank_text, candidates_4_rank_gt_text])
            
            demo_elems.append(demo_elem[::-1])

        return demo_elems


    def predict_on_subsets(self, interaction, idxs, valid_data, selected_uids):

        demo_elems = self.demo_construction(valid_data, selected_uids)

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
            demo_elem= demo_elems[i]
            input_ = self.prompt_func(self.dataset_name, user_his_text, candidate_text_order,
                                              self.recall_budget, demo_elem)

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
                    self.logger.info(f'Ground-truth [{target_text}]: Ranks {ground_truth_pr}')
                    self.logger.info(f'Ranking List [{rec_item_idx_list}]')
                except:
                    self.logger.info(f'Fail to find ground-truth items.')
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

        prompt_merge_path = osp.join(self.data_path, "output_files",
                                     f"final11_{self.api_model_name}_{self.dataset_name}_{self.config.model}_output_{self.version}_{self.config['seed']}_{self.config['rep_num']}.json")
        with open(prompt_merge_path, mode='w') as f:
            json.dump(result_dict, f, indent=2)
        self.logger.info(f"SAVE DONE! Path: {prompt_merge_path}")

        if self.boots:
            scores = scores.view(self.boots, -1, scores.size(-1))
            scores = scores.sum(0)
        return scores 
