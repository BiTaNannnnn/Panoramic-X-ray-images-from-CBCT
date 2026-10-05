"""
Author: XXX
Date: Nov 2019
"""
import argparse
import os
from torch.nn.utils import clip_grad_norm_
import time
from PIL import Image
from torch.utils.data import DataLoader, random_split
from PointNet_3d.data_utils.S3DISDataLoader import S3DISDataset
from PointNet_3d.data_utils.ToothLoader import ToothLoader, evaluate_iou
from PointNet_3d.models.Unet_sem_seg import UNet, PointNet_3d, Feature_Net
import torch
import torch.nn as nn
import datetime
import logging
from pathlib import Path
import torch.nn.functional as F
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
from utils.dice_score import dice_loss, FocalLoss
import wandb

# dir
out_filename = Path('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log/visual/0117_CBAM_all_Visial/')
out_filename.mkdir(exist_ok=True)

# 3d
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + '/PointNet_3d'
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def parse_args():
    # 3d paser
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='Unet_sem_seg', help='model name [default:Unet_sem_seg pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=900000, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='Initial learning rate [default: 0.001]') # default=5.0E-5,1e-5
    parser.add_argument('--gpu', type=str, default='5', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')  # RMS （seg net）
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    # seg parser
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data3d that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=48, help='Number of classes')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_generate')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # '''Load the pretrained UNet model'''
    # # pretrained_unet = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    # state_dict = torch.load('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/checkpoints/Seg_1211_512/checkpoint_epoch2100.pth', map_location=device)
    # # pretrained_unet.to(device=device)
    # # Remove the "mask_values" key from the state_dict
    # mask_values = state_dict.pop('mask_values', [0, 1])
    # # pretrained_unet.load_state_dict(state_dict)
    # # # 冻结 U-Net 模型
    # # for param in pretrained_unet.parameters():
    # #     param.requires_grad = False

    # '''Load the pretrained UNet model'''

    logging.info('Pretrain Model loaded!')

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)


    n_classes = args.classes
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    IMG_SCALE = 1.0
    val_percent: float = 0.02
    dir_img = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_padding_512_test3/imgs/' # Data_Tooth  train_test3;train_padding_512_test3
    dir_mask = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_padding_512_test3/masks/'
    dir_GT = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/GT/'

    print("start loading training data ...")
    # TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    DATASET = ToothLoader(dir_img=dir_img, dir_mask=dir_mask, dir_GT=dir_GT, num_point=NUM_POINT, img_scale=IMG_SCALE, block_size=1.0, transform=None)

    print("start loading test data ...")

    # 2. Split into train / validation partitions
    n_val = int(len(DATASET) * val_percent)
    n_train = len(DATASET) - n_val
    TRAIN_DATASET, TEST_DATASET = random_split(DATASET, [n_train, n_val], generator=torch.Generator().manual_seed(0))


    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
    #                                               pin_memory=True, drop_last=True,
    #                                               worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    # shutil.copy('PointNet_3d/models/%s.py' % args.model, str(experiment_dir))
    # shutil.copy('PointNet_3d/models/pointnet2_utils.py', str(experiment_dir))
    # Unet_sem_seg.py --get_model -- get loss
    classifier = MODEL.get_model(n_classes, n_channels=3,  bilinear=False).cuda()
    '''Initialize your UNetPoint model with pretrained UNet weights'''
    # classifier.unet_part.load_state_dict(state_dict, strict=False)
    criterion = MODEL.get_loss().cuda()
    criterion_seg = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()

    # criterion = torch.nn.MSELoss().cuda()    # L2
    # criterion = criterion.to('cuda')
    # print(criterion.device)
    classifier.apply(inplace_relu)
    # (Initialize logging)

    ''' 分别为 U-Net 和 PointNet 定义参数组'''
    # # 定义不同部分的学习率和权重衰减
    unet_params = {'params': classifier.unet_part.parameters(), 'lr': 1e-5, 'weight_decay': 0.0005}
    # pointnet_params = {'params': classifier.pointnet_part.parameters()}
    # feature_params =  {'params': classifier.feature_part.parameters()}

    experiment = wandb.init(project='PointNet-3D-All', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=args.epoch, batch_size=BATCH_SIZE, learning_rate=args.learning_rate, val_percent=0.1)
    )

    # def weights_init(m):
    #     classname = m.__class__.__name__
    #     # if classname.find('Conv2d') != -1:
    #     if classname.find('Conv1d') != -1:
    #         torch.nn.init.xavier_normal_(m.weight.data)
    #         torch.nn.init.constant_(m.bias.data, 0.0)
    #     elif classname.find('Linear') != -1:
    #         torch.nn.init.xavier_normal_(m.weight.data)
    #         torch.nn.init.constant_(m.bias.data, 0.0)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv1d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 1
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        # optimizer = torch.optim.Adam([unet_params, pointnet_params])
        optimizer = torch.optim.Adam(
            [
                {'params': classifier.unet_part.parameters(), 'lr': 1e-5, 'weight_decay': 0.0005},
                # {'params': classifier.pointnet_part.parameters()},
                # {'params': classifier.feature_part.parameters()},
            ],
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == 'RMS':
        optimizer = torch.optim.RMSprop(
            [
                {'params': classifier.unet_part.parameters(), 'lr': 1e-5, 'weight_decay': 0.0005},
                # {'params': classifier.pointnet_part.parameters()},
                # {'params': classifier.feature_part.parameters()},
            ],
            lr=args.learning_rate,  # 这里的 lr 是默认学习率，对于没有单独设置 lr 的参数组有效
            weight_decay=args.decay_rate,
            momentum=0.999,
            foreach=True
        )

    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 1
    best_iou = 0
    start_vram = torch.cuda.memory_allocated()
    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch, epoch, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)

        loss_sum = 0
        classifier = classifier.train()
        # # 记录读取数据之前的时间
        # start_time = time.time()
        with tqdm(total=len(trainDataLoader), desc=f'Epoch {epoch}/{args.epoch}', unit='pts') as pbar:
            for batch in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
                optimizer.zero_grad()
                ID = batch['ID'][0]
                images = batch['image']
                true_masks = batch['mask']
                points = batch['points']
                # change
                # points = points.squeeze(0)  # (28,4096,3)
                target = batch['GT_points']
                mask_values = batch['mask_values']
                mask_original = batch['mask_original']
                # print('patient ID = ', ID)
                # print('patient tooth sum = ', len(mask_values))

                # # check shape of input
                # # 保存为 .obj 文件
                # points_input = points[0]
                # obj_filename = os.path.join(str(out_filename) + '/input_shape_ball.obj')
                # with open(obj_filename, 'w') as f:
                #     for point in points_input:
                #         f.write(f'v {point[0]} {point[1]} {point[2]}\n')

                # # 可视化gt  target
                # vertices = target[235].squeeze(0).numpy()
                # # 创建一个 PointCloud 对象
                # cloud = trimesh.points.PointCloud(vertices)
                # # 将结果保存为 PLY 文件
                # cloud.export('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log/visual/gt_visual.ply')

                assert images.shape[1] == classifier.n_channels, \
                    f'Network has been defined with {classifier.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                images = images.cuda()
                true_masks = true_masks.cuda()
                mask_original = mask_original.cuda()
                points = points.float().cuda()
                points = points.transpose(2, 1)
                # target dic into cuda
                for key in target.keys():
                    target[key] = target[key].cuda()
                # # 计算并打印读取数据所需的时间
                # print("Time taken to load data: ", time.time() - start_time)

                # start_time_model = time.time()
                # model Entrance
                logits, seg_feature, tooth_outputs = classifier(images, mask_values, mask_original, points)  # (16,4096,3)
                # # 计算并打印训练网络所需的时间
                # print("Time taken to train network: ", time.time() - start_time_model)

                # # save pre img
                masks_pred = logits
                # savepath = Path(str(out_filename) + '/' + str(ID))
                # savepath.mkdir(exist_ok=True)
                # masks_pred_copy = masks_pred.cpu().argmax(dim=1).numpy()
                # all_mask_values = np.asarray(mask_original.cpu().squeeze().numpy())
                # all_mask_values = np.unique(np.concatenate(all_mask_values), axis=0)
                # result = mask_to_image(masks_pred_copy, all_mask_values)
                # result.save(str(savepath) + '/PreMask.png')
                # # unique = np.unique(result)
                # # print(unique)
                if ID == 'CASE_ID':  # CASE_ID
                    masks_pred_copy = masks_pred.cpu().argmax(dim=1).numpy()
                    # all_mask_values = np.asarray(mask_original.cpu().squeeze().numpy())
                    # all_mask_values = np.unique(np.concatenate(all_mask_values), axis=0)
                    all_mask_values = [0, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 55, 60, 65, 70, 75, 80, 85, 90, 105, 110, 115,
                     120, 125, 130, 135, 140, 155, 160, 165, 170, 175, 180, 185, 190, 205, 210, 215, 220, 225, 230, 235, 240]
                    # result = masks_pred_copy.squeeze()
                    result = mask_to_image(masks_pred_copy, all_mask_values)
                    result.save(str(out_filename) + '/PreMask.png')

                # 记录loss之前的时间
                # start_time_loss = time.time()
                num_teeth = len(target)  # change
                # num_teeth = 1
                total_loss = 0
                for tooth in tooth_outputs.keys():
                    # 计算每颗牙齿的损失
                    tooth_output = tooth_outputs[tooth]  # 获取每颗牙齿的输出
                    seg_pred = tooth_output['x']  # 1，4096，3：第tooth颗牙齿
                    if ID == 'CASE_ID':  # CASE_ID
                        # 遍历每个点云+ save points    # 获取第一颗牙齿的点云 seg_pred[0]的形状是(4096, 3)
                        points = seg_pred[0]
                        # 保存为 .obj 文件
                        with open(str(out_filename) + '/' + ID + '_teeth_' + str(int(tooth)) + '.obj', 'w') as f:
                            for point in points:
                                f.write(f'v {point[0]} {point[1]} {point[2]}\n')

                    target_teeth = target[int(tooth)]
                    tooth_loss = criterion(seg_pred, target_teeth)  # 计算损失
                    total_loss += tooth_loss

                # log_string('Saving tooth obj at %s' % out_filename)

                # 计算平均损失
                loss = total_loss / num_teeth

                # seg loss
                loss_seg = dice_loss(
                    F.softmax(masks_pred, dim=1).float(),
                    F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
                loss_seg += FocalLoss(masks_pred, true_masks, n_classes, device)

                # loss_seg = criterion_seg(masks_pred, true_masks)
                # loss_seg += dice_loss(
                #     F.softmax(masks_pred, dim=1).float(),
                #     F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float(),
                #     multiclass=True
                # )
                loss_sum = loss + loss_seg

                # # 计算并打印loss所需的时间
                # print("Time taken to calculate loss network: ", time.time() - start_time_loss)

                # 反向传播计算梯度
                loss_sum.backward()
                # 检查梯度是否包含 NaN 值
                grad_nan = False
                for name, param in classifier.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        grad_nan = True
                        break

                # 如果梯度中不包含 NaN 值，则更新模型参数
                if not grad_nan:
                    optimizer.step()
                # 更新梯度
                # optimizer.step()

                experiment.log({
                    '3d train loss': loss.item(),
                    'segmentation loss ': loss_seg.item(),
                    'ALL loss == ': loss_sum.item(),
                    'step': global_epoch,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'3d loss (batch)': loss.item()})
                pbar.set_postfix(**{'seg loss (batch)': loss_seg.item()})
                pbar.set_postfix(**{'ALL loss (batch)': loss_sum.item()})
                # 更新进度
                pbar.update(1)
                # 检查梯度 change
                # check_model_gradients(classifier)

                # pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
                # correct = np.sum(pred_choice == batch_label)
                # total_correct += correct
                # total_seen += (BATCH_SIZE * NUM_POINT)
                # loss_sum += loss
            log_string('Training SEG MEAN loss: %f' % loss_seg)
            log_string('Training 3D  MEAN loss: %f' % loss)
            log_string('Training ALL MEAN loss: %f' % loss_sum)
            # log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

            # # 计算并打印读取数据所需的时间
            # print("Time taken to train one epoch: ", time.time() - start_time)
            end_vram = torch.cuda.memory_allocated()
            vram_usage = (end_vram - start_vram) / (1024 ** 3)  # 转换为GB
            print(f"Training on an RTX 3090 for {vram_usage:.3f}GB VRAM")
            if epoch % 10 == 0:
                logger.info('Save model...')
                # 模型文件名包含轮数信息
                model_path = f'/model_epoch_{epoch}.pth'

                savepath = str(checkpoints_dir) + model_path
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')



            '''Evaluate all on chopped scenes   write by XXX'''
            with torch.no_grad():
                num_batches = len(testDataLoader)

                with tqdm(total=len(testDataLoader), desc=f'Epoch {epoch}/{args.epoch }', unit='pts') as pbar:
                    for batch in tqdm(testDataLoader, total=len(testDataLoader), smoothing=0.9):
                        # input points
                        images = batch['image']
                        true_masks = batch['mask']
                        points = batch['points']
                        # change
                        # points = points.squeeze(0)
                        target = batch['GT_points']
                        mask_values = batch['mask_values']
                        mask_original = batch['mask_original']
                        assert images.shape[1] == classifier.n_channels, \
                            f'Network has been defined with {classifier.n_channels} input channels, ' \
                            f'but loaded images have {images.shape[1]} channels. Please check that ' \
                            'the images are loaded correctly.'
                        images = images.cuda()
                        true_masks = true_masks.cuda()
                        mask_original = mask_original.cuda()
                        points = points.float().cuda()
                        points = points.transpose(2, 1)
                        # target dic into cuda
                        for key in target.keys():
                            target[key] = target[key].cuda()

                        # model Entrance
                        logits, seg_feature, tooth_outputs = classifier(images, mask_values, mask_original,
                                                                        points)
                        # seg output
                        masks_pred = logits
                        # loss + iou
                        num_teeth = len(target)
                        total_loss = 0
                        total_iou = 0
                        for tooth in tooth_outputs.keys():
                            # 计算每颗牙齿的损失
                            tooth_output = tooth_outputs[tooth]  # 获取每颗牙齿的输出
                            seg_pred = tooth_output['x']
                            # GT
                            target_teeth = target[int(tooth)]
                            tooth_loss = criterion(seg_pred, target_teeth)  # 计算损失
                            total_loss += tooth_loss
                            # IOU
                            iou = evaluate_iou(seg_pred, target_teeth)
                            total_iou += iou

                        ################
                        # 计算3D平均损失
                        loss = total_loss / num_teeth
                        # 计算seg loss
                        loss_seg = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        loss_seg += FocalLoss(masks_pred, true_masks, n_classes, device)


                        # loss_seg = criterion_seg(masks_pred, true_masks)
                        # loss_seg += dice_loss(
                        #     F.softmax(masks_pred, dim=1).float(),
                        #     F.one_hot(true_masks, n_classes).permute(0, 3, 1, 2).float(),
                        #     multiclass=True
                        # )

                        # ALL mean loss
                        loss_sum = loss + loss_seg

                        # 计算 平均 IoU 值 mean
                        mean_iou = total_iou / num_teeth
                        mean_iou_scalar = torch.mean(mean_iou)
                        print(f'Epoch: {epoch}, Mean_IoU: {mean_iou_scalar.item():.4f}')

                        log_string('\n#################   eval 3D mean loss: %f' % loss)
                        log_string('\n#################   eval SEG mean loss: %f' % loss_seg)
                        log_string('\n#################   eval ALL mean loss: %f' % loss_sum)
                        # print(f'Epoch: {epoch}, Mean_IoU: {mean_iou.item():.4f}')
                        pbar.set_postfix(**{'3d loss (batch)': loss})
                        pbar.set_postfix(**{'seg loss (batch)': loss_seg})
                        pbar.set_postfix(**{'ALL loss (batch)': loss_sum})
                        # 更新进度
                        # pbar.update(1)


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
    mask = mask.squeeze()

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)
    else:
        pass
    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def check_model_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"Gradient of parameter {name} contains NaN values")
            if torch.isinf(param.grad).any():
                print(f"Gradient of parameter {name} contains infinite values")


if __name__ == '__main__':
    args = parse_args()
    main(args)
