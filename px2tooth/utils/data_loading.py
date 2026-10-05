import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import RandomVerticalFlip, RandomHorizontalFlip, RandomRotation


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        mask = mask.flatten()
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


def get_bbox(mask, padding=0):
    mask_gray = np.max(mask, axis=-1)
    # 找到mask中所有非零元素的坐标
    rows, cols = np.where(mask_gray != 0)
    # 计算边界框
    top = max(0, np.min(rows) - padding)
    bottom = min(mask_gray.shape[0], np.max(rows) + padding)
    left = max(0, np.min(cols) - padding)
    right = min(mask_gray.shape[1], np.max(cols) + padding)
    return top, bottom, left, right


def center_crop(img, mask, bbox, crop_size=(200, 340)):
    top, bottom, left, right = bbox
    bbox_width = right - left
    bbox_height = bottom - top
    # 计算裁剪区域的中心点
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    # 如果bbox的宽度大于crop_size的宽度，那么在bbox的宽度范围内随机选择一个起始点
    if bbox_width > crop_size[1]:
        start_x = np.random.randint(left, right - crop_size[1] + 1)
    else:
        # 如果bbox的宽度不大于crop_size的宽度，那么在bbox的中心周围随机选择一个起始点
        max_dx = min(center_x - left, right - center_x, crop_size[1] // 2 - bbox_width // 2)
        if max_dx > 0:
            dx = np.random.randint(-max_dx, max_dx)
        else:
            dx = 0
        start_x = max(0, left + dx)

    # 如果bbox的高度大于crop_size的高度，那么在bbox的高度范围内随机选择一个起始点
    if bbox_height > crop_size[0]:
        start_y = np.random.randint(top, bottom - crop_size[0] + 1)
    else:
        # 如果bbox的高度不大于crop_size的高度，那么在bbox的中心周围随机选择一个起始点
        max_dy = min(center_y - top, bottom - center_y, crop_size[0] // 2 - bbox_height // 2)
        if max_dy > 0:
            dy = np.random.randint(-max_dy, max_dy)
        else:
            dy = 0
        start_y = max(0, bottom + dy)
    # 计算裁剪区域的结束点
    end_x = start_x + crop_size[1]
    end_y = start_y + crop_size[0]
    # 检查是否超出图像边界
    if end_x > img.shape[1]:
        start_x -= end_x - img.shape[1]
        end_x = img.shape[1]
    if end_y > img.shape[0]:
        start_y -= end_y - img.shape[0]
        end_y = img.shape[0]
    # 裁剪图像和mask
    img_crop = img[start_y:end_y, start_x:end_x]
    mask_crop = mask[start_y:end_y, start_x:end_x]
    return img_crop, mask_crop



class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0,
                 mask_suffix: str = '_mask'):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        # with Pool() as p:
        #     unique = list(tqdm(
        #         p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
        #         total=len(self.ids)
        #     ))
        unique = []
        for idx in self.ids:
            result = unique_mask_values(idx, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix)
            unique.append(result)

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        # img_values = np.unique(img)
        # print('img_values = ', img_values)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            # mask_values_one = np.unique(mask)
            # print('mask_values_one = ', mask_values_one)
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    @staticmethod
    def prodece(image):
        image = np.array(image)
        if image.shape[0] > 512:  # crop
            p = image.shape[0] // 2 - 256
            image = image[p:p + 512, :, :]
        elif image.shape[0] < 512:  # padding
            image_tmp = np.full([512, image.shape[1], image.shape[-1]], fill_value=0, dtype=image.dtype)
            p = 256 - image.shape[0] // 2
            l = image.shape[0]
            image_tmp[p:p + l, :, :] = image
            image = image_tmp

        #  h
        if image.shape[1] > 512:  # crop
            p = image.shape[1] // 2 - 256
            image = image[:, p:p + 512, :]
        elif image.shape[1] < 512:  # padding
            image_tmp = np.full([512, 512, image.shape[-1]], fill_value=0, dtype=image.dtype)
            p = 256 - image.shape[1] // 2
            l = image.shape[1]
            image_tmp[:, p:p + l, :] = image
            image = image_tmp
        image = Image.fromarray(image)
        return image


    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        ######################## 定义随机裁剪变换 方法
        # # 结果：最小宽度: 341, 最小高度: 200 (height, width)
        # transform = transforms.Compose([
        #     transforms.RandomCrop((200, 340)),
        # ])

        # # 剪裁+ 翻转
        # transform = transforms.Compose([
        #     RandomVerticalFlip(),
        #     RandomHorizontalFlip(),
        #     RandomRotation(degrees=90),
        #     transforms.RandomCrop((200, 340)),
        # ])
        # 获取一个随机种子
        # seed = torch.random.seed()
        # # 设置随机种子
        # torch.random.manual_seed(seed)
        # # 对图像进行裁剪
        # img = transform(img)
        #
        # # 为了保证图像和对应的标签裁剪的区域相同，我们需要再次设置相同的随机种子
        # torch.random.manual_seed(seed)
        # # 对标签进行裁剪
        # mask = transform(mask)
        ########################

        # ######################## crop box 周围大小方法
        # img = np.asarray(img)
        # mask = np.asarray(mask)
        # bbox = get_bbox(mask)
        # img, mask = center_crop(img, mask, bbox)
        # if mask.shape != (200, 340, 3):
        #     print("mask的形状不是(200, 340, 3)，而是{}".format(mask.shape))
        # ########################


        # # test
        # img_save = Image.fromarray(img)
        # mask_save = Image.fromarray(mask)
        # img_save.save('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_130/'+ name +'.png')
        # mask_save.save('xxx/Project/3D_Dental_Master/Pytorch-UNet-master/Data_Tooth/train_130/' + name + '_mask.png')

        # img = Image.fromarray(img)
        # mask = Image.fromarray(mask)
        img = self.prodece(img)
        mask = self.prodece(mask)
        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)
        voxel_grid_unique = np.unique(mask)
        # print(voxel_grid_unique)
        ############################


        # # print(img.shape)
        # # print(mask.shape)
        # # print(np.unique(mask))
        # # exit()
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'ID': name
        }





class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='')
