# LLMSRec_Syn
Code for Our NAACL Findings 2024 paper "The Whole is Better than the Sum: Using Aggregated Demonstrations in In-Context Learning for Sequential Recommendation"

## 🚀 Quick Start

1. Unzip dataset files.
    ```bash
    cd LLMSRec_Syn/dataset/ml-1m/; unzip ml-1m.inter.zip
    cd LLMSRec_Syn/dataset/Games/; unzip Games.inter.zip
    ```
    For data preparation details, please refer to LLMRank's [[data-preparation]](https://github.com/RUCAIBox/LLMRank/blob/master/llmrank/dataset/data-preparation.md).
2. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
3. Insert your OpenAI key on line 20 of ./model/rank.py.
4. Evaluate OpenAI GPT (GPT-3.5-turbo)'s ranking abilities on ml-1m (Games/lastfm) dataset.
    ```bash
    cd LLMSRec_Syn/
    python run_test.py -m RankAggregated(ours)/RankNearest(baseline)/RankFiexed(baseline) -d ml-1m
    ```

## Results

### ml-1m, gpt-3.5-turbo
RankAggregated 
```bash
INFO  SAVE DONE! Path: dataset/ml-1m/output_files/final11_gpt-3.5-turbo_ml-1m_RankAggregated_output_2020.json
INFO  
recall@1 - 0.265  recall@5 - 0.6  recall@10 - 0.775       recall@20 - 1.0 recall@50 - 1.0 
ndcg@1 - 0.265  ndcg@5 - 0.4424 ndcg@10 - 0.4988    ndcg@20 - 0.5549        ndcg@50 - 0.5549```
```

RankNearest
```bash
INFO  SAVE DONE! Path: dataset/ml-1m/output_files/final11_gpt-3.5-turbo_ml-1m_RankNearest_output_2020.json
INFO  
recall@1 - 0.26   recall@5 - 0.6  recall@10 - 0.76        recall@20 - 1.0 recall@50 - 1.0 
ndcg@1 - 0.26   ndcg@5 - 0.4356 ndcg@10 - 0.4867    ndcg@20 - 0.5465        ndcg@50 - 0.5465
```

RankFixed
```bash
INFO  SAVE DONE! Path: dataset/ml-1m/output_files/final11_gpt-3.5-turbo_ml-1m_RankFixed_output_2020.json
INFO  
recall@1 - 0.195  recall@5 - 0.58 recall@10 - 0.76        recall@20 - 1.0 recall@50 - 1.0 
ndcg@1 - 0.195  ndcg@5 - 0.3969 ndcg@10 - 0.4539    ndcg@20 - 0.5138        ndcg@50 - 0.5138
```

## 🌟 Cite Us

Please cite the following paper if you find our code helpful.

```bibtex
@article{wang2024whole,
  title={The Whole is Better than the Sum: Using Aggregated Demonstrations in In-Context Learning for Sequential Recommendation},
  author={Wang, Lei and Lim, Ee-Peng},
  journal={arXiv preprint arXiv:2403.10135},
  year={2024}
}
```

The experiments are conducted using the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole) and [LLMRank](https://github.com/RUCAIBox/LLMRank).
