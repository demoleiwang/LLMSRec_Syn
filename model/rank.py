import os.path as osp
import torch
import openai
from openai import AsyncOpenAI
import numpy as np 
import time
import asyncio
import numpy as np
from tqdm import tqdm
from recbole.model.abstract_recommender import SequentialRecommender
from sklearn.metrics.pairwise import cosine_similarity
from utils import dispatch_openai_requests, dispatch_single_openai_requests
import re
# import jsonlines
import json
import os
import random
import httpx

openai_api_key = ""
openai_client = AsyncOpenAI(
    api_key=openai_api_key,
    # base_url = "https://api.xi-ai.cn/v1"
    # http_client=httpx.AsyncClient(proxies="https://api.xi-ai.cn/v1")
)

async def handle_request(message, model, temperature, max_tokens):
    # Your logic to handle individual requests
    response = await openai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=message,
        )
    return response

async def dispatch_openai_requests(messages_list, model, temperature, max_tokens=512):
    # See https://platform.openai.com/docs/guides/gpt for details
    async_responses = [handle_request(message, model, temperature, max_tokens) for message in messages_list]
    response_contents = await asyncio.gather(*async_responses)
    return response_contents


class Rank(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.max_tokens = config['max_tokens']
        # openai.api_base = "https://api.xi-ai.cn/v1"
        self.api_model_name = self.config['platform']
        openai.api_key = self.config["key"]

        self.api_batch = config['api_batch']
        self.async_dispatch = config['async_dispatch']
        self.temperature = config['temperature']

        self.max_his_len = config['max_his_len']
        self.recall_budget = config['recall_budget']
        self.boots = config['boots']
        self.data_path = config['data_path']
        self.dataset_name = dataset.dataset_name
        self.id_token = dataset.field2id_token['item_id']
        self.item_text = self.load_text()
        self.logger.info(f'Avg. t = {np.mean([len(_) for _ in self.item_text])}')

        self.fake_fn = torch.nn.Linear(1, 1)

    def load_text(self):
        token_text = {}
        item_text = ['[PAD]']
        feat_path = osp.join(self.data_path, f'{self.dataset_name}.item')
        if self.dataset_name in ['ml-1m', 'ml-01m']:
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, movie_title, release_year, genre = line.strip().split('\t')
                    token_text[item_id] = movie_title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                if raw_text.endswith(', The'):
                    raw_text = 'The ' + raw_text[:-5]
                elif raw_text.endswith(', A'):
                    raw_text = 'A ' + raw_text[:-3]
                item_text.append(raw_text)
            return item_text
        elif self.dataset_name == 'Games':
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, title = line.strip().split('\t')
                    token_text[item_id] = title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                item_text.append(raw_text)
            return item_text
        elif self.dataset_name == 'lastfm':
            with open(feat_path, 'r', encoding='utf-8') as file:
                file.readline()
                for line in file:
                    item_id, title = line.strip().split('\t')
                    token_text[item_id] = title
            for i, token in enumerate(self.id_token):
                if token == '[PAD]': continue
                raw_text = token_text[token]
                item_text.append(raw_text)
            return item_text
        else:
            raise NotImplementedError()

    
    def get_batch_inputs(self, interaction, idxs, i):
        user_his = interaction[self.ITEM_SEQ]
        user_his_len = interaction[self.ITEM_SEQ_LEN]
        origin_batch_size = user_his.size(0)
        real_his_len = min(self.max_his_len, user_his_len[i % origin_batch_size].item())
        user_his_text = [str(j) + '. ' + self.item_text[user_his[i % origin_batch_size, user_his_len[i % origin_batch_size].item() - real_his_len + j].item()] \
                for j in range(real_his_len)]
        candidate_text = [self.item_text[idxs[i,j]]
                for j in range(idxs.shape[1])]
        candidate_text_order = [str(j) + '. ' + self.item_text[idxs[i,j].item()]
                for j in range(idxs.shape[1])]
        candidate_idx = idxs[i].tolist()

        return user_his_text, candidate_text, candidate_text_order, candidate_idx

    def parsing_output_text(self, scores, i, response_list, idxs, candidate_text):
        rec_item_idx_list = []
        found_item_cnt = 0
        for j, item_detail in enumerate(response_list):
            if len(item_detail) < 1:
                continue
            if item_detail.endswith('candidate movies:'):
                continue
            pr = item_detail.find('. ')
            if item_detail[:pr].isdigit():
                item_name = item_detail[pr + 2:]
            else:
                item_name = item_detail

            matched_name = None
            for candidate_text_single in candidate_text:
                if candidate_text_single in item_name:
                    if candidate_text_single in rec_item_idx_list:
                        break
                    rec_item_idx_list.append(candidate_text_single)
                    matched_name = candidate_text_single
                    break
            if matched_name is None:
                continue

            candidate_pr = candidate_text.index(matched_name)
            scores[i, idxs[i, candidate_pr]] = self.recall_budget - found_item_cnt
            found_item_cnt += 1
        return rec_item_idx_list

    def parsing_output_indices(self, scores, i, response_list, idxs, candidate_text):
        rec_item_idx_list = []
        found_item_cnt = 0
        for j, item_detail in enumerate(response_list):
            if len(item_detail) < 1:
                continue

            if not item_detail.isdigit():
                continue

            pr = int(item_detail)
            if pr >= self.recall_budget:
                continue
            matched_name = candidate_text[pr]
            if matched_name in rec_item_idx_list:
                continue
            rec_item_idx_list.append(matched_name)
            scores[i, idxs[i, pr]] = self.recall_budget - found_item_cnt
            found_item_cnt += 1
            if len(rec_item_idx_list) >= self.recall_budget:
                break

        return rec_item_idx_list

    
    def dispatch_openai_api_requests(self, prompt_list, batch_size):
        openai_responses = []
        self.logger.info('Launch OpenAI APIs')

        # if self.async_dispatch:
        self.logger.info('Asynchronous dispatching OpenAI API requests.')
        for i in tqdm(range(0, batch_size, self.api_batch)):
            openai_responses += asyncio.run(
                        dispatch_openai_requests(prompt_list[i:i+self.api_batch], self.api_model_name, self.temperature)
                    )
            # while True:
            #     try:
            #         openai_responses += asyncio.run(
            #             dispatch_openai_requests(prompt_list[i:i+self.api_batch], self.api_model_name, self.temperature)
            #         )
            #         break
            #     except KeyboardInterrupt:
            #         print(f'KeyboardInterrupt Error, retry batch {i // self.api_batch} at {time.ctime()}', flush=True)
            #         time.sleep(20)
            #     except Exception as e:
            #         print(f'Error {e}, retry batch {i // self.api_batch} at {time.ctime()}', flush=True)
            #         time.sleep(20)
        self.logger.info('Received OpenAI Responses')
        return openai_responses


    def get_similarity_matrix(self):

        if self.config['data_name'] == "Games":

            train_emb_file = os.path.join(self.config['data_path'], "output_files", f"train_multivector_emb.npy")
            test_emb_file = os.path.join(self.config['data_path'], "output_files", f"test_multivector_emb.npy")

            normalized_train_user_matrix = np.load(train_emb_file)
            normalized_test_user_matrix = np.load(test_emb_file)

            user_matrix_aug_sim = cosine_similarity(normalized_test_user_matrix, normalized_train_user_matrix)

            possible_ids = []
            for test_ui in range(200):
                sim_per_xx_ids = np.argsort(-user_matrix_aug_sim[test_ui])
                sample_demo_ids = list(sim_per_xx_ids[0:15])
                possible_ids.extend(sample_demo_ids)
            possible_ids = set(possible_ids)
            unpossible_ids = []
            for xxixx in range(user_matrix_aug_sim.shape[-1]):
                if xxixx not in possible_ids:
                    unpossible_ids.append(xxixx)
            unpossible_ids = np.array(sorted(list(unpossible_ids)))
            possible_ids = np.array(sorted(list(possible_ids)))
            user_matrix_aug_sim[:, unpossible_ids] = 0.


        train_emb_file = os.path.join(self.config['data_path'], "output_files", f"train_{self.config['sim']}_emb.npy")
        test_emb_file = os.path.join(self.config['data_path'], "output_files", f"test_{self.config['sim']}_emb.npy")

        normalized_train_user_matrix = np.load(train_emb_file)
        normalized_test_user_matrix = np.load(test_emb_file)

        if self.config['data_name'] == "Games":
            user_matrix_aug_sim_x = cosine_similarity(normalized_test_user_matrix, normalized_train_user_matrix)
            user_matrix_aug_sim[:,possible_ids] = user_matrix_aug_sim_x
        else:
            user_matrix_aug_sim = cosine_similarity(normalized_test_user_matrix, normalized_train_user_matrix)
        return user_matrix_aug_sim

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