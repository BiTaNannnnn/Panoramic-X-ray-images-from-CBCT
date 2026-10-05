"""
Author: XXX
Date: Nov 2019
"""
import argparse
import os
from data_utils.S3DISDataLoader import ScannetDatasetWholeScene
from data_utils.indoor3d_util import g_label2color
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import provider
import numpy as np


# 3d
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='xxx/Project/3D_Dental_Master/PointNet_3d/log',
                        help='experiment root')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=int, default=5, help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=3,
                        help='aggregate segmentation scores with voting [default: 5]')
    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # experiment_dir = 'xxx/Project/3D_Dental_Master/PointNet_3d/log/sem_seg/2023-05-27_02-35'
    experiment_dir = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log/sem_seg/2023-06-27_09-06'
    visual_dir = experiment_dir + '/visual/'
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    NUM_CLASSES = 13
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    # root = 'data/s3dis/stanford_indoor3d/'
    #
    # TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    # log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        points = torch.rand(16, 4096, 3)
        points = points.squeeze(0)
        points = points.float().cuda()
        points = points.transpose(2, 1)
        seg_pred, _ = classifier(points)
        seg_pred = seg_pred.cpu().numpy()
        savepath = 'xxx/Project/3D_Dental_Master/PointNet_3d/predict_mesh/'

        # 遍历每个点云
        for i in range(seg_pred.shape[0]):
            # 获取当前点云
            points = seg_pred[i]
            # 保存为 .obj 文件
            with open(savepath + 'output1_{i}.obj', 'w') as f:
                for point in points:
                    f.write(f'v {point[0]} {point[1]} {point[2]}\n')

        print('save path = ', savepath)
        print("Done!")
        # output_2d = seg_pred.reshape(-1, 3)
        # # 使用 numpy.savetxt 函数保存数组
        # np.savetxt(savepath, output_2d)


if __name__ == '__main__':
    args = parse_args()
    main(args)
