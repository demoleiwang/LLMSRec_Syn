# LLMSRec_Syn (Updating)
Code for Our NAACL Findings 2024 paper "The Whole is Better than the Sum: Using Aggregated Demonstrations in In-Context Learning for Sequential Recommendation"

## 🚀 Quick Start

1. Write your own OpenAI API keys into [`LLMSRec_Syn/openai_api.yaml`](https://github.com/demoleiwang/LLMSRec_Syn/blob/master/llmrank/openai_api.yaml).
2. Unzip dataset files.
    ```bash
    cd LLMSRec_Syn/dataset/ml-1m/; unzip ml-1m.inter.zip
    cd LLMSRec_Syn/dataset/Games/; unzip Games.inter.zip
    ```
    For data preparation details, please refer to LLMRank's [[data-preparation]](https://github.com/RUCAIBox/LLMRank/blob/master/llmrank/dataset/data-preparation.md).
3. Install dependencies.
    ```bash
    pip install -r requirements.txt
    ```
4. Evaluate OpenAI GPT (GPT-4o)'s ranking abilities on ml-1m dataset.
    ```bash
    cd LLMSRec_Syn/
    python evaluate.py -m RankAggregated(ours)/RankNearest(baseline)/RankFiexed(baseline)/RankZero(Our zero-shot version) -d ml-1m
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
