"""
Author: XXX
测试TrainAll 的测试集
Date: Nov 2019
"""
import argparse
from PointNet_3d.data_utils.ToothLoader import *
from utils.IOU import *
import logging
from pathlib import Path
import sys
import importlib
import torchvision.transforms as transforms
from utils.evaluation_metrics import *

# seg
# dir_img = Path('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_padding_512/imgs/') # Data_Tooth
# dir_mask = Path('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_padding_512/masks/')


# 3d
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR + '/PointNet_3d'
sys.path.append(os.path.join(ROOT_DIR, 'models'))



def parse_args():
    # 3d paser
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='Unet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=900000, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=5.0E-5, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='6', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
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
    '''CREATE DIR'''
    experiment_dir = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log'
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

    n_classes = args.classes
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    IMG_SCALE = 1.0
    val_percent: float = 0.1

    # root = 'data/s3dis/stanford_indoor3d/'
    #
    # TEST_DATASET_WHOLE_SCENE = ScannetDatasetWholeScene(root, split='test', test_area=args.test_area, block_points=NUM_POINT)
    # log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    '''MODEL LOADING'''
    # file_name = os.listdir(experiment_dir + '/sem_generate')[0]
    file_name = '2024-01-09_PSA_TA_alldata'
    model_name = os.listdir(experiment_dir + '/sem_generate/'+file_name+'/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(n_classes, n_channels=3,  bilinear=False).cuda()
    model_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log/sem_generate/2024-01-17_CrossAttention/checkpoints/model_epoch_170.pth'
    # model_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log/sem_generate/2023-09-25_MeshSize2800Lun/checkpoints/model_epoch_2800.pth'
    checkpoint = torch.load(model_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    # classifier = classifier.eval()

    iou_list = []
    iou_3dList= []
    MMD_cdList = []
    MMD_emdList = []
    COV_cdList = []
    COV_emdList = []

    with torch.no_grad():
        # points 方法一 = 正方体均匀取点
        n = 16
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        z = torch.linspace(0, 1, n)
        xx, yy, zz = torch.meshgrid(x, y, z)
        points = torch.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], dim=1)
        points = points.unsqueeze(0)

        # points 方法二 = 正方体随机取点
        # points = torch.rand(1, 4096, 3)

        # # points 方法三 = 球表面积均匀取点
        # points = fibonacci_sphere(samples=4096)
        # points = points.unsqueeze(0)

        # # check shape
        # # 保存为 .obj 文件
        # points_input = points[0]
        # save_path = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log/visual/CASE_ID_0704/'
        # obj_filename = os.path.join(save_path + 'input_shape_random.obj')
        # with open(obj_filename, 'w') as f:
        #     for point in points_input:
        #         f.write(f'v {point[0]} {point[1]} {point[2]}\n')

        # change
        points = points.float().cuda()
        points_in = points.transpose(2, 1)
        # imgs + masks
        transform = transforms.ToTensor()
        dir_img_file = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_padding_512/test/'  # Data_Tooth;train_512*512size
        dir_mask_file = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_padding_512/test_masks/'
        dir_GT_file = 'xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/GT/'

        for root, dirs, files in os.walk(dir_img_file):
            for file in files:
                # 获取文件的完整路径
                ID_name = file.split('.')[0]
                print(ID_name)
                dir_img = dir_img_file + ID_name+'.png'
                dir_mask = dir_mask_file+ID_name+'_mask.png'
                mask_original = Image.open(dir_mask)
                true_mask = mask_original
                # show_result(true_mask, 'Seg Pre Mask')
                # mask_values
                all_mask_values = np.asarray(mask_original)
                all_mask_values = all_mask_values[:, :, 0]
                all_mask_values = np.unique(np.concatenate(all_mask_values), axis=0)


                non_zero_values = all_mask_values[all_mask_values != 0]
                mask_values = list(sorted(non_zero_values.tolist()))
                mask_values = [torch.tensor(x) for x in mask_values if x >= 50]
                values_to_remove = [240, 190, 140, 90]  # 需要移除的值 保留28颗牙齿
                for value in values_to_remove:
                    while value in mask_values:
                        mask_values.remove(value)
                # to tensor to cuda()
                mask_original = np.asarray(mask_original)
                mask_original = torch.as_tensor(mask_original.copy()).long().contiguous()
                unique_mask = np.unique(mask_original)
                print(unique_mask)
                mask_original = mask_original.unsqueeze(0)
                '''img'''
                all_values = [0, 11, 12, 13, 14, 21, 22, 23, 24, 31, 32, 33, 41, 42, 43, 45, 55, 60, 65, 70, 75, 80, 85,
                              90, 105, 110, 115, 120, 125, 130, 135, 140, 155, 160, 165, 170, 175, 180, 185, 190, 205, 210, 215, 220, 225,
                              230, 235, 240]
                images = Image.open(dir_img)
                images = preprocess(all_values, images, scale=1.0, is_mask=False)
                images = images.astype(np.float32)
                images = torch.from_numpy(images)
                images = images.cuda().unsqueeze(0)
                # # show_result(images, 'Seg Pre Mask')
                # images = transform(images).float().contiguous()
                # # unique = np.unique(images)
                # # print(unique)

                # GT points
                # 读入一个id 的牙齿mesh
                mesh_path = dir_GT_file + ID_name
                target = read_teeth_mesh(mesh_path)
                # target dic into cuda
                for key in target.keys():
                    target[key] = torch.tensor(target[key]).cuda()

                # model Entrance
                logits, seg_feature, tooth_outputs = classifier(images, mask_values, mask_original, points_in)  # (16,4096,3)

                masks_pred = logits.cpu().argmax(dim=1).numpy().squeeze()
                # masks_pred = masks_pred[0].cpu().long().squeeze().numpy()
                unique = np.unique(masks_pred)
                print(unique)
                # _PreMask + save
                # seg_pred = masks_pred.long().cpu()
                # masks_pred = masks_pred.argmax(dim=1).numpy()
                # mask = mask[0].squeeze()
                Date = '0117_CrossAttention_170_Infer'
                filename = os.path.join(visual_dir, Date)
                filename = check_dir(filename)
                out_filename = os.path.join(filename, str(ID_name)+'/')
                out_filename = check_dir(out_filename)
                # all 48 values

                result = mask_to_image(masks_pred, all_values)
                # show_result(result, 'Seg Pre Mask')
                # show_result(mask_original, 'Seg label GT')
                # iou
                print('ID_name = ', ID_name)
                result.save(out_filename + 'PreMask.png')
                iou = mIOU(true_mask, result)
                iou_list.append(iou)
                unique = np.unique(result)
                print(unique)
                logging.info(f'Mask saved to {out_filename}')
                print(f'Mask saved to {out_filename}')

                # PrePoints + save
                tooth_points = []
                total_iou = 0
                MMD_cd = 0
                MMD_emd = 0
                COV_cd = 0
                COV_emd = 0

                # 遍历每个点云
                for tooth in tooth_outputs.keys():
                    # 获取当前点云
                    # tooth = 235
                    # teeth_point = tooth_outputs[tooth]['x'].cpu().numpy().squeeze(0)
                    teeth_point = tooth_outputs[tooth]['x']
                    # 遍历每个点云
                    for i in range(teeth_point.shape[0]):
                        # 获取当前点云
                        points = teeth_point[i]
                        # 保存为 .obj 文件
                        obj_filename = os.path.join(out_filename + f'teeth_{tooth}.obj')
                        with open(obj_filename, 'w') as f:
                            for point in points:
                                f.write(f'v {point[0]} {point[1]} {point[2]}\n')

                    # teeth_point = torch.mean(teeth_point, dim=0, keepdim=True)
                    # teeth_point = teeth_point.cpu().numpy().squeeze(0)
                    # # 保存为 .obj 文件
                    # obj_filename = os.path.join(out_filename + f'teeth_{tooth}.obj')
                    # with open(obj_filename, 'w') as f:
                    #     for point in teeth_point:
                    #         f.write(f'v {point[0]} {point[1]} {point[2]}\n')
                    # # 计算俩点云 iou
                    target_teeth = target[int(tooth)]
                    target_teeth = target_teeth.clone().detach()
                    '''计算IOU-3D'''
                    iou = evaluate_iou(teeth_point, target_teeth)
                    total_iou += iou
                    # print(str(tooth), ".iou = ", iou)
                    '''计算 MMD-CD/EMD  and COV-CD/EMD'''
                    MMD_cd, MMD_emd, COV_cd, COV_emd = compute_metrics(teeth_point, target_teeth)
                    MMD_cd += MMD_cd
                    MMD_emd += MMD_emd
                    COV_cd += COV_cd
                    COV_emd += COV_emd
                    tooth_points.append(teeth_point)

                # 把每颗牙齿 加起来成为整个口腔牙齿
                # go on
                mean_iou = total_iou/len(tooth_outputs)
                mean_MMD_cd = MMD_cd/len(tooth_outputs)
                mean_MMD_emd = MMD_emd/len(tooth_outputs)
                mean_COV_cd = COV_cd/len(tooth_outputs)
                mean_COV_emd = COV_emd/len(tooth_outputs)

                iou_3dList.append(mean_iou)
                MMD_cdList.append(mean_MMD_cd)
                MMD_emdList.append(mean_MMD_emd)
                COV_cdList.append(mean_COV_cd)
                COV_emdList.append(mean_COV_emd)

                print("ALL-ID-TOOTH ..3Dmean_iou = ", mean_iou)
                print("ALL-ID-TOOTH ..mean_MMD_cd = ", mean_MMD_cd)
                print("ALL-ID-TOOTH ..mean_MMD_emd = ", mean_MMD_emd)
                print("ALL-ID-TOOTH ..mean_COV_cd = ", mean_COV_cd)
                print("ALL-ID-TOOTH ..mean_COV_emd = ", mean_COV_emd)
                print("--------------------Done!-----------------")
    # mIoU pre
    print("mIoU List = ", iou_list)
    print("iou_3dList = ", iou_3dList)
    # print("MMD_cdList = ", MMD_cdList)
    # print("MMD_emdList = ", MMD_emdList)
    # print("COV_cdList = ", COV_cdList)
    # print("COV_emdList = ", COV_emdList)

    iou_3d_average = sum(iou_3dList) / len(iou_3dList)
    MMD_cd_average = sum(MMD_cdList) / len(MMD_cdList)
    MMD_emd_average = sum(MMD_emdList) / len(MMD_emdList)
    COV_cd_average = sum(COV_cdList) / len(COV_cdList)
    COV_emd_average = sum(COV_emdList) / len(COV_emdList)

    print("Average value of 3D_mIoUList: ", iou_3d_average)
    print("Average value of MMD_cdList: ", MMD_cd_average)
    print("Average value of MMD_emdList: ", MMD_emd_average)
    print("Average value of COV_cdList: ", COV_cd_average)
    print("Average value of COV_emdList: ", COV_emd_average)


    input_array = np.array(iou_list)
    average = np.mean(input_array)
    print("Average of 2DmIoU List:", average)




def fibonacci_sphere(samples=4096):
    points = np.zeros((samples, 3))
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y
        theta = phi * i  # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points[i] = np.array([x, y, z])
        # to tensor
    points = torch.from_numpy(points)
    return points


def check_dir(dir_path, INFO=False):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print('Making new directory: %s' % dir_path)
    return dir_path


def show_result(image, title=None):
    image = np.flip(image, axis=0)
    plt.figure()
    plt.imshow(image, cmap='gray')
    if title:
        plt.title(title)
    plt.show()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def preprocess(mask_values, pil_img, scale, is_mask):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img = np.asarray(pil_img)

    if is_mask:
        mask = np.zeros((newH, newW), dtype=np.int64)
        for i, v in enumerate(mask_values):
            if img.ndim == 2:
                mask[img == v] = i
            else:
                mask[(img == v).all(-1)] = i
        return mask

    else:
        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img


if __name__ == '__main__':
    args = parse_args()
    main(args)

