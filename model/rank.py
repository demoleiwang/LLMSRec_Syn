import os.path as osp
import torch
import openai
from openai import OpenAI
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

openai_api_key = "sk-0lpH9abgDoJQLAkwCf4e64Bf88004229B1481e4f8dF9C0E5"
openai_client = OpenAI(
    api_key=openai_api_key,
)



async def dispatch_openai_requests(messages_list, model, temperature, max_tokens=512):
    # See https://platform.openai.com/docs/guides/gpt for details
    # Chat completions API
    response = [
        openai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=mesasage,
        )
        for message in messages_list
    ]
    return await asyncio.gather(*async_responses) #response.choices[0].message.content


class Rank(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.max_tokens = config['max_tokens']
        self.api_model_name = config['api_name']
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

        if self.async_dispatch:
            self.logger.info('Asynchronous dispatching OpenAI API requests.')
            for i in tqdm(range(0, batch_size, self.api_batch)):
                while True:
                    try:
                        openai_responses += asyncio.run(
                            dispatch_openai_requests(prompt_list[i:i+self.api_batch], self.api_model_name, self.temperature)
                        )
                        break
                    except KeyboardInterrupt:
                        print(f'KeyboardInterrupt Error, retry batch {i // self.api_batch} at {time.ctime()}', flush=True)
                        time.sleep(20)
                    except Exception as e:
                        print(f'Error {e}, retry batch {i // self.api_batch} at {time.ctime()}', flush=True)
                        time.sleep(20)
        else:
            self.logger.info('Dispatching OpenAI API requests one by one.')
            for message in tqdm(prompt_list):
                openai_responses.append(dispatch_single_openai_requests(message, self.api_model_name, self.temperature))
                # print (openai_responses)
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

