import argparse
import logging
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from PIL import Image
import wandb
from evaluate import evaluate
from unet.unet_model import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss,    FocalLoss

dir_img = Path('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_all/imgs/') # Data_Tooth train_512*512size// train_all
dir_mask = Path('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_all/masks/')
dir_checkpoint = Path('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/checkpoints/Seg_0516/')
out_filename = Path('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log/visual/Seg_0516')

# 如果amp=True，那么就启用混合精度训练
def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.02,  #0.3,
        save_checkpoint: bool = True,
        img_scale: float = 1.0,
        amp: bool = False,
        weight_decay: float = 1e-4,  # 1e-8
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)  #  num_workers=os.cpu_count()
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    log_string(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    # 创建损失函数，忽略值为-1
    # criterion = nn.CrossEntropyLoss(ignore_index=-1)

    global_step = 0


    log_string(args)

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks, ID = batch['image'], batch['mask'], batch['ID']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    if ID == 'CASE_ID':  # CASE_ID
                        masks_pred_copy = masks_pred.cpu().argmax(dim=1).numpy()
                        all_mask_values = np.asarray(true_masks.copy().cpu().squeeze().numpy())
                        all_mask_values = np.unique(np.concatenate(all_mask_values), axis=0)
                        result = mask_to_image(masks_pred_copy, all_mask_values)
                        result.save(str(out_filename) + 'CASE_ID-PreMask.png')

                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        # # 设置要忽略的值 mask 已经映射成连续的values
                        # ignore_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
                        #
                        # # 将 true_masks 中等于这些值的元素都设置为 -1。
                        # for val in ignore_values:
                        #     true_masks[true_masks == val] = -1
                        # test_mask1 = copy.deepcopy(true_masks)

                        # 原始loss
                        # loss = criterion(masks_pred, true_masks)
                        # loss += dice_loss(
                        #     F.softmax(masks_pred, dim=1).float(),
                        #     F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #     multiclass=True
                        # )

                        # 组合loss
                        # loss = basnet_hybrid_loss(masks_pred, true_masks, model.n_classes)
                        # loss_fn = FocalTverskyLoss()
                        # loss = loss_fn(masks_pred, true_masks)
                        # loss += dice_loss(
                        #         F.softmax(masks_pred, dim=1).float(),
                        #         F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #         multiclass=True
                        #     )

                        # boundary_loss = BoundaryDoULoss(model.n_classes)
                        # loss = boundary_loss(masks_pred, true_masks)
                        #
                        # loss = criterion(masks_pred, true_masks)
                        # loss += boundary_loss(masks_pred, true_masks)

                        # Tloss
                        # Tloss_loss = TLoss(model.parameters(), device, nu=1.0, epsilon=1e-8)
                        # loss = Tloss_loss(masks_pred, true_masks)

                        # # JDTLoss
                        # JDTLoss_loss = JDTLoss(alpha=0.5, beta=0.5)
                        # loss = criterion(masks_pred, true_masks)
                        # loss += JDTLoss_loss(masks_pred, true_masks)

                        # PolyLoss_loss = PolyLoss()
                        # loss = criterion(masks_pred, true_masks)
                        loss = dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        loss += FocalLoss(masks_pred, true_masks, model.n_classes, device)

                        log_string(f'loss = {loss}')

                optimizer.zero_grad(set_to_none=True)  # 迭代开始时，手动梯度清零：backward()梯度会累积
                grad_scaler.scale(loss).backward()  # 缩放损失 反向传播
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)  # 对模型梯度进行剪裁，按比例缩小梯度，防止梯度爆炸
                # 器中所有参数的梯度进行反缩放，如果这些梯度不包含无穷大或NaN值，则调用optimizer.step()来更新参数；否则会跳过参数更新步骤。
                grad_scaler.step(optimizer)
                # 更新缩放因子以准备下一次迭代 如果在上一次迭代中发现了无穷大或NaN的梯度 那么缩放因子会被减小以防止下一次迭代再次出现这种情况
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                # division_step = 10
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        # for tag, value in model.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     if not (torch.isinf(value) | torch.isnan(value)).any():
                        #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                        #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        mIOU, dice_score, evl_loss = evaluate(model, val_loader, device, amp)
                        scheduler.step(dice_score)

                        log_string(f'evl_loss = {evl_loss}')

                        log_string('Validation Dice score: {}'.format(dice_score))
                        log_string('mIOU score: {}'.format(mIOU))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': dice_score,
                                'mIOU': mIOU,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass


        if epoch % 20 == 0 and save_checkpoint:
        # if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            log_string(f'Checkpoint {epoch} saved!')



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', type=str, default='Unet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=900000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,  # 0 。1=1e-5
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=48, help='Number of classes')

    return parser.parse_args()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)
    mask = mask.squeeze()
    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

if __name__ == '__main__':
    args = get_args()
    # if CUDA is available
    train_on_gpu = torch.cuda.is_available()
    if train_on_gpu:
        print('CUDA is available, Training on GPU ...')
    else:
        print('CUDA is not available!  Training on CPU ...')


    # + log save
    def log_string(str):
        logger.info(str)
        print(str)


    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)

    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')  # 数字切换卡号
    log_string(f'Using device {device}')


    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    log_string(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        log_string(f'Model loaded from {args.load}')


    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=0.02,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=0.02,
            amp=args.amp
        )
