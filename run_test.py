import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from utils import get_model
from trainer import SelectedUserTrainer
from openai_parallel_toolkit import ParallelToolkit, Prompt


def evaluate(model_name, dataset_name, pretrained_file, version, rep_num, num_data, platform, random_seed, **kwargs):
    # configurations initialization
    props = [f'props/{model_name}.yaml', f'props/{dataset_name}.yaml', 'props/overall.yaml']
    print(props)
    model_class = get_model(model_name)

    # configurations initialization
    config = Config(model=model_class, dataset=dataset_name, config_file_list=props, config_dict=kwargs)
    config['seed'] = random_seed
    init_seed(config['seed'], config['reproducibility'])
    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    logger.info(f"{config['seed']}, reproducibility: {config['reproducibility']}")

    dataset = SequentialDataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    config['platform'] = platform
    model = model_class(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.load_other_parameter(checkpoint.get("other_parameter"))

    # variants
    config['rep_num'] = rep_num
    if model_name not in ['SASRec'] and  version != '':
        model.assign_version(version)

    logger.info(model)

    # trainer loading and initialization
    config['num_data'] = num_data

    trainer = SelectedUserTrainer(config, model, dataset, version)

    ##### for zero-shot, acl 2024 short.
    if model_name in ["SASRec", "GRU4Rec", "BERT4Rec", "Pop"]:
        test_result = trainer.evaluatex(test_data, load_best_model=False, show_progress=config['show_progress'])
    else:
        test_result = trainer.evaluate(test_data, load_best_model=False, show_progress=config['show_progress'])

    # init_logger(config)
    # logger = getLogger()
    # logger.info(set_color('test result', 'yellow') + f': {test_result}')
    output_res = []
    for u, v in test_result.items():
        output_res.append(f'{u} - {v}')
    logger.info('\t'.join(output_res))

    return config['model'], config['dataset'], {
        'valid_score_bigger': config['valid_metric_bigger'],
        'test_result': test_result
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=str, default="Rank", help="model name")
    parser.add_argument('-d', type=str, default='ml-1m', help='dataset name') # [ml-1m, lastfm, Games]
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')
    parser.add_argument('-v', type=str, default='rank_aggregated', help='method name')
    # [rank_aggregated, rank_fixed, rank_nearest, rank_zero]

    parser.add_argument('-r', type=str, default='0', help='repeat number')
    parser.add_argument('-n', type=int, default=5000, help='repeat number')

    parser.add_argument('-pl', type=str, default="gpt-4o", help='openai engine') # [gpt-4o, gpt-3.5-turbo]
    parser.add_argument('-sd', type=int, default=2020, help='')

    args, unparsed = parser.parse_known_args()
    print(args)

    evaluate(args.m, args.d, pretrained_file=args.p, version=args.v, rep_num=args.r, num_data=args.n, platform=args.pl, random_seed=args.sd)
