import argparse
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.utils import init_seed, init_logger, get_trainer, set_color
from utils import get_model
from trainer import SelectedUserTrainer


def evaluate(model_name, dataset_name, pretrained_file, **kwargs):
    # configurations initialization
    props = [f'props/{model_name}.yaml', f'props/{dataset_name}.yaml', 'props/overall.yaml']
    print(props)
    model_class = get_model(model_name)

    # configurations initialization
    config = Config(model=model_class, dataset=dataset_name, config_file_list=props, config_dict=kwargs)
    
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
    model = model_class(config, train_data.dataset).to(config['device'])

    # Load pre-trained model
    if pretrained_file != '':
        checkpoint = torch.load(pretrained_file)
        logger.info(f'Loading from {pretrained_file}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.load_other_parameter(checkpoint.get("other_parameter"))

    logger.info(model)

    trainer = SelectedUserTrainer(config, model, dataset)

    trainer.get_emb_multivector(train_data, valid_data, test_data, load_best_model=False, show_progress=config['show_progress'])

    test_result = trainer.evaluate(test_data, valid_data, load_best_model=False, show_progress=config['show_progress'])

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
    parser.add_argument("-m", type=str, default="RankZero", help="model name")
    # [RankAggregated, RankFixed, RankNearest, RankZero]

    parser.add_argument('-d', type=str, default='ml-1m', help='dataset name') # [ml-1m, lastfm, Games]
    parser.add_argument('-p', type=str, default='', help='pre-trained model path')

    parser.add_argument('-n', type=int, default=200, help='number of ')

    parser.add_argument('-pl', type=str, default="gpt-3.5-turbo", help='openai engine') # [gpt-4o, gpt-3.5-turbo]
    # parser.add_argument('-key', type=str, default="", help='openai key') # [
    parser.add_argument('-sd', type=int, default=2020, help='')

    args, unparsed = parser.parse_known_args()
    print(args)

    config_dict = {
        "platform": args.pl,
        "seed": args.sd,
        'num_demo_int': 2,
        'num_demo_out': 1,
        'sim': "multivector", # openaiemb
        'num_data': args.n
    }

    evaluate(model_name=args.m, dataset_name=args.d, pretrained_file=args.p, **config_dict)
