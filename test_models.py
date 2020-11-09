import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str


import cv2

import numpy as np
import torch
from basicsr.models.base_model import BaseModel
from basicsr.models.sr_model import SRModel


img = cv2.imread('my_images/input/interpolate_0000.png')
#img2 = np.expand_dims(img, 0)
img2 = np.rollaxis(img, 2, 0)
img3 = torch.from_numpy(img2)

basis_opt = {
    'num_gpu': 0,
    'is_train': False
}

basis_model = BaseModel(basis_opt)


sr_opt = {i: j for i, j in basis_opt.items()}
sr_opt['dist'] = False
sr_opt['network_g'] = {
    'type': 'EDSR',
    'num_in_ch': 3,
    'num_out_ch': 3,
    'num_feat': 256,
    'num_block': 32,
    'upscale': 2,
    'res_scale': 0.1,
    'img_range': 255.,
    'rgb_mean': [0.4488, 0.4371, 0.4040]
}
sr_opt['path'] = {
    'pretrain_network_g': 'experiments/pretrained_models/EDSR/EDSR_Lx2_f256b32_DIV2K_official-be38e77d.pth',
    'strict_load_g': True
}

sr_model = SRModel(sr_opt)
sr_model.net_g(img3)

sr_result = sr_model.net_g(img3)
sr_img = sr_result.detach().numpy()[0,:]
sr_img = np.rollaxis(sr_img, 0, 3)

cv2.imwrite('my_images/sr.png', sr_img)


def main():
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed'])
        logger.info(
            f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)

    # create model
    model = create_model(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(
            test_loader,
            current_iter=opt['name'],
            tb_logger=None,
            save_img=opt['val']['save_img'])


if __name__ == '__main__':
    main()
