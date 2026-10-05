from model.base_model import *

#resnet
class AutoEncoder(BaseModel):
    def __init__(self, dataset, network, model_name='AutoEncoder', MPR=False, cuda_id=0):
        super(AutoEncoder, self).__init__(dataset, network, model_name, MPR, cuda_id)
        self.optimizers = {'optim': torch.optim.Adam(self.network.parameters(), lr=0.001, weight_decay=0.001)}
        # 0.0001 for 3D

    def train(self, start_epoch=0, epoch_n=100):
        # load saved ckpt if start_epoch != 0
        if start_epoch:
            self._load_ckpt(start_epoch)

        sampler = self.dataset.train_sampler

        self.best_score = self.val(start_epoch, self.dataset.val_sampler, SAVE=False)
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
                if score > self.best_score:
                    self.best_score = score
                    self._save_ckpt(epoch_id, BEST=True)