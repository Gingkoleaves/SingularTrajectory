from VaeRepresentationNet import *
import os
import argparse
import baseline

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # 添加当前目录到 Python 路径

from utils import *
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default="./config/singulartrajectory-transformerdiffusion-zara1.json", type=str, help="config file path")
    parser.add_argument('--tag', default="SingularTrajectory-TEMP", type=str, help="personal tag for the model")
    parser.add_argument('--gpu_id', default="0", type=str, help="gpu id for the model")
    parser.add_argument('--test', default=False, action='store_true', help="evaluation mode")
    args = parser.parse_args()

    print("===== Arguments =====")
    print_arguments(vars(args))

    print("===== Configs =====")
    hyper_params = get_exp_config(args.cfg)
    print_arguments(hyper_params)

    dataset_dir = hyper_params.dataset_dir + hyper_params.dataset + '/'
    obs_len, pred_len, skip = hyper_params.obs_len, hyper_params.pred_len, hyper_params.skip
    
    loader_train = get_dataloader(dataset_dir, 'train', obs_len, pred_len, batch_size=1, skip=skip)
    loader_val = get_dataloader(dataset_dir, 'val', obs_len, pred_len, batch_size=1)
    loader_test = get_dataloader(dataset_dir, 'test', obs_len, pred_len, batch_size=1)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    vae_trainer = VaeTrainer(hyper_params)
    # Initialize the VAE trainer
    for epoch in range(1, 129):
        vae_trainer.train(epoch)

    for loss in vae_trainer.log['train_loss']:
        print(f"Train Loss: {loss:.4f}")

    