from model.base_model import *
# from R2N2_utils.layers import SoftmaxWithLoss3D
import torch


class R2N2_Encoder(BaseModel):
    def __init__(self, dataset, network):
        super(R2N2_Encoder, self).__init__(dataset, network, MPR=False, model_name='R2N2_Encoder', cuda_id=0)
        self.optimizers = {'optim': torch.optim.Adam(self.network.parameters(), lr=0.001)}
        self.network.cuda()
        # self.softmax_loss = SoftmaxWithLoss3D()
        # 0.0001 for 3D

    def train(self, start_epoch=0, epoch_n=100):
        # load saved ckpt if start_epoch != 0
        if start_epoch:
            self._load_ckpt(start_epoch)
        sampler = self.dataset.train_sampler
        # self.best_score = self.val(start_epoch, self.dataset.val_sampler, SAVE=False)
        optim = self.optimizers['optim']
        start_epoch += 1
        for epoch_id in range(start_epoch, start_epoch + epoch_n):
            optim = self._adjust_learning_rate(optim, epoch_id, 0.001)
            with tqdm(total=sampler.data_n) as epoch_pbar:
                loss_list = []
                for batch_id in range(sampler.batch_n):
                    batch = sampler.get_batch()
                    input_tensor = torch.tensor(batch['Ideal_PX'], dtype=torch.float)
                    gt_tensor = torch.tensor(batch['MPR'], dtype=torch.float) if self.MPR \
                        else torch.tensor(batch['CBCT'], dtype=torch.float)

                    if self.cuda_id is not None:
                        input_tensor = input_tensor.cuda(self.cuda_id)
                        gt_tensor = gt_tensor.cuda(self.cuda_id)
                    # update generator
                    optim.zero_grad()
                    loss, generations_MPR = self.inference(input_tensor, y=gt_tensor)
                    loss.backward()
                    optim.step()
                    loss_list.append(loss.data.cpu().numpy())
                    # update training info
                    desc = f'Epoch:{epoch_id:04d}|loss {loss:.4f}'
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(input_tensor.shape[0])

                desc = f'Epoch:{epoch_id:04d}|loss {np.mean(loss_list):.4f}'
                epoch_pbar.set_description(desc)
                epoch_pbar.close()

            # validation
            SAVE = epoch_id % self.save_n == 0
            if SAVE:
                self._save_ckpt(epoch_id, BEST=False)
            if epoch_id % self.val_n == 0:
                score = self.val(epoch_id, self.dataset.val_sampler, SAVE)
                # if score > self.best_score:
                #     self.best_score = score
                #     self._save_ckpt(epoch_id, BEST=True)

    # def train(self, start_epoch, epoch_n):
    #     if start_epoch:
    #         self._load_ckpt(start_epoch)
    #
    #     sampler = self.dataset.train_sampler
    #     optim = self.optimizers['optim']
    #     # initialize the start point
    #     # self.best_loss = self.val(start_epoch, self.dataset.val_sampler)
    #     start_epoch += 1
    #     for epoch_id in range(start_epoch, start_epoch + epoch_n):
    #         optim = self._adjust_learning_rate(optim, epoch_id, 0.001)
    #         loss_list = []
    #         with tqdm(total=sampler.data_n) as epoch_pbar:
    #             for batch_id in range(sampler.batch_n):
    #                 batch = sampler.get_batch()
    #                 # batch_img = self.compact_views(batch['Ideal_PX'])
    #                 input_tensor = torch.tensor(batch['Ideal_PX'], dtype=torch.float).cuda()
    #                 gt_tensor = torch.tensor(batch['CBCT'], dtype=torch.float).cuda()
    #
    #                 output = self.network(input_tensor)
    #                 loss = torch.nn.functional.mse_loss(output, gt_tensor)
    #                 optim.zero_grad()
    #                 loss.backward()
    #                 optim.step()
    #
    #                 loss = loss.data3d.cpu().numpy()
    #                 loss_list.append(loss)
    #                 desc = f'Epoch:{epoch_id:04d}|loss {loss:.4f}'
    #                 epoch_pbar.set_description(desc)
    #                 epoch_pbar.update(input_tensor.shape[1])
    #
    #             desc = f'Epoch:{epoch_id:04d}|loss {np.mean(loss_list):.4f}'
    #             epoch_pbar.set_description(desc)
    #             epoch_pbar.close()
    #         # validation
    #         SAVE = epoch_id % self.save_n == 0
    #         if SAVE:
    #             self._save_ckpt(epoch_id, BEST=False)
    #         if epoch_id % self.val_n == 0:
    #             loss = self.val(epoch_id, self.dataset.val_sampler, SAVE)
    #             # if loss < self.best_loss:
    #             #     self.best_loss = loss
    #             #     self._save_ckpt(epoch_id, BEST=True)

    def inference(self, input_tensor, y, prior_shapes=None):
        # if the mode is only inference
        if y is None:
            generations = self.network(input_tensor).detach()
            generations_cpu = generations.data.cpu().numpy()
            # if the shape is provided, registrant the image back to the old space
            if prior_shapes:
                generation_list = []
                for generation_cpu, prior_shape in zip(generations_cpu, prior_shapes):
                    generation_list.append(interpolation(generation_cpu, prior_shape))
                generations_cpu = generation_list
            return generations_cpu

        # if the mode is training
        else:
            # generations = self.network(px_tensor)
            # loss = torch.nn.functional.mse_loss(generations, y)
            generations = self.network(input_tensor)
            loss = torch.nn.functional.mse_loss(generations, y)

            return loss, generations

    def val(self, epoch_id, sampler, SAVE=False, mode='Val', EVAL=True):
        psnr_list = []
        ssim_list = []
        dice_list = []
        for batch_id in range(sampler.batch_n):
            # sample batch
            batch = sampler.get_batch()
            batch_img = self.compact_views(batch['Ideal_PX'])
            batch_size = np.shape(batch_img)[1]
            input_tensor = torch.tensor(batch_img, dtype=torch.float).cuda()
            generations = self.inference(input_tensor)
            for item_id in range(batch_size):
                generation = generations[item_id]
                if EVAL:
                    # evaluate performance
                    CBCT = batch['CBCT'][item_id]
                    Bone = batch['Bone'][item_id]
                    MPR = batch['MPR'][item_id]
                    batch_shape = batch['PriorShape']

                    psnr_list.append(get_psnr(generation, CBCT))
                    dice_list.append(get_dice(generation > -0.8, interpolation(MPR, batch_shape[item_id]) > -0.5))
                    ssim_list.append(self.ssim_funct.eval_ssim(generation, CBCT))

                # save results
                if SAVE:
                    case_id = batch['Case_ID'][item_id]
                    if mode == 'Val':
                        generation_dir = check_dir(join_path(self.result_dir, mode + '_%d' % epoch_id))
                    elif mode == 'Test':
                        generation_dir = check_dir(join_path(self.result_dir, mode + '_%d' % self.dataset.TEST_MODE))
                    else:
                        raise ValueError('Unknown validation mode: %s' % mode)
                    generation_img = np.array((generation + 1) * 2000, dtype=np.uint16)
                    save_nii(generation_img, np.eye(4), 'case_%03d.nii.gz' % case_id, generation_dir)
        if EVAL:
            # calculate average value
            PSNR = np.mean(psnr_list)
            SSIM = np.mean(ssim_list)
            Dice = np.mean(dice_list)
            PSNR_std = np.std(psnr_list)
            SSIM_std = np.std(ssim_list)
            Dice_std = np.std(dice_list)
            avg_score = (PSNR / 20 + SSIM + Dice) / 3 * 100

            title = mode + '_%d' % epoch_id
            desc = f'{title}|PSNR {PSNR:.4f} {PSNR_std:.4f}, SSIM {SSIM:.4f} {SSIM_std:.4f}, IOU {IOU:.4f} {IOU_std:.4f}'
            print(colored(desc, 'blue'))
            return 1 - SSIM

    @staticmethod
    def compact_views(batch_img):
        img = batch_img[0][0, :, :]
        # multiview_img = np.zeros(shape=[3, 1, 1, 160, 288])
        multiview_img = np.zeros(shape=[3, 1, 1, 440, 288])
        multiview_img[0, 0, 0, :, :] = img[:, 0: 288]
        multiview_img[1, 0, 0, :, :] = img[:, 144: 144+288]
        multiview_img[2, 0, 0, :, :] = img[:, 288:]
        return multiview_img / 256

    @staticmethod
    def downscale_gt(gt_tensor):
        gt_tensor = gt_tensor.unsqueeze(0)
        return torch.nn.Upsample(scale_factor=0.5)(gt_tensor)