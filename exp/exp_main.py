from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
# from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST, SegRNN, CycleNet, ZWF修改loss版, \
#     iTransformer, TimeXer, TQNet, TQDLinear, TQPatchTST, TQiTransformer, ZWF, ZWF_TQNet_1 , ZWF_TQNet_2
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import logging
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.use_miss = args.use_miss
        self.enc_in = args.enc_in

        self.logger = logging.getLogger(f"logger_{args.logger_uique_id}")
        self.args = args

    def _build_model(self):
        try:
            # 动态导入 models 文件夹下与模型名同名的模块
            import importlib
            module = importlib.import_module(f"models.{self.args.model}")
            # 从模块里取 Model 类
            model_class = getattr(module, "Model")
            model = model_class(self.args).float()
            print(f"✅Successfully loaded model {self.args.model}.")
        except ModuleNotFoundError:
            # 模块不存在时，使用默认模型
            print("❌Warning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"❌Warning: model {self.args.model} not found. Using default ZWF_TQNet_1.")
            raise ModuleNotFoundError(f"Model {self.args.model} not found in models folder.")
            # default_module = importlib.import_module("models.ZWF_TQNet_1")
            # model_class = getattr(default_module, "Model")
            # model = model_class(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        logger = logging.getLogger(f"logger_{self.args.logger_uique_id}")
        logger.info("模型总的参数数量: {:.4f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):

        self.model.mode_now = "vali"
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(vali_loader):
                # 1. 数据统一迁移至设备 (修复了原代码 batch_y 漏掉 to(device) 的问题)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # 2. 构建 Decoder 输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 💡 3. 构建全能参数字典
                model_kwargs = {
                    "x_enc": batch_x,
                    "x_mark_enc": batch_x_mark,
                    "x_dec": dec_inp,
                    "x_mark_dec": batch_y_mark,
                    "batch_y": batch_y,
                    "cycle_index": batch_cycle
                }

                additional_loss = None

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(**model_kwargs)

                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        if self.args.output_attention:
                            # 假设需要输出 attention 的模型返回 (outputs, attention_weights)
                            outputs = outputs[0]
                        else:
                            # 带额外 Loss 的模型返回 (outputs, additional_loss)
                            outputs, additional_loss = outputs[0], outputs[1]



                # 5. 截取特征维度与预测长度
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # 6. 转回 CPU 供后续计算 Validation Loss 记录
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if np.isnan(pred.numpy()).any() :
                    print("Warning: Pred contains NaN values!")
                if np.isnan(true.numpy()).any():
                    print("Warning: True contains NaN values!")

                loss = criterion(pred, true)
                if additional_loss is not None:
                    if hasattr(additional_loss, 'detach'):
                        additional_loss = additional_loss.detach().cpu().item() 
                    else:
                        # 如果已经是 float 了，就直接保持原样
                        additional_loss = float(additional_loss)
                    loss = loss + additional_loss


                # print("loss的类型", type(loss), "loss的值", loss.item())
                if np.isnan(loss.item()).any():
                    print("Warning: Loss is NaN!")

                total_loss.append(loss)
        # print(f"Validation losses are: {[l.item() for l in total_loss]}")  # 打印每个批次的损失值
        total_loss = np.average(total_loss)
        if np.isnan(total_loss).any():
            print("Warning: Total validation loss is NaN!")
        self.model.train() # 调整模型回到训练模式
        return total_loss

    def train(self, setting):
        
        self.model.mode_now = "train"

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(logger_uique_id=self.args.logger_uique_id, patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                # 1. 数据迁移至设备
                batch_x = batch_x.float().to(self.device)               
                batch_y = batch_y.float().to(self.device)               
                batch_x_mark = batch_x_mark.float().to(self.device)     
                batch_y_mark = batch_y_mark.float().to(self.device)     
                batch_cycle = batch_cycle.int().to(self.device)         

                # 2. 构建 Decoder 输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                model_kwargs = {
                    "x_enc": batch_x,
                    "x_mark_enc": batch_x_mark,
                    "x_dec": dec_inp,
                    "x_mark_dec": batch_y_mark,
                    "batch_y": batch_y,  # 模型需要的真实标签
                    "cycle_index": batch_cycle  # CycleNet 和 TQNet 需要的周期索引
                }
                
                additional_loss = None

                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
            
                    outputs = self.model(**model_kwargs)
                    
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        if self.args.output_attention:
                            # 假设需要输出 attention 的模型返回 (outputs, attention_weights)
                            outputs = outputs[0]
                        else:
                            # 带额外 Loss 的模型返回 (outputs, additional_loss)
                            outputs, additional_loss = outputs[0], outputs[1]

                    # 6. 截取预测部分与特征维度
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_target = batch_y[:, -self.args.pred_len:, f_dim:]

                    # 7. 计算 Loss
                    loss = criterion(outputs, batch_y_target)
                    if additional_loss is not None:
                        if torch.is_tensor(additional_loss):
                            loss = loss + additional_loss.to(self.device)
                        else:
                            # 如果是 float，直接加就行，不需要 to(device)
                            loss = loss + additional_loss

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    self.logger.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    self.logger.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()

                    scaler.unscale_(model_optim)  # AMP 模式下必须先 unscale
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)  # 梯度裁剪
                    # if total_norm > self.args.max_norm:
                    #     print(f"梯度裁剪 Total norm before clipping: {total_norm:.4f}")
                    scaler.scale(loss)

                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()

                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),  self.args.max_norm)
                    # if total_norm >  self.args.max_norm:
                    #     print(f"梯度裁剪 Total norm before clipping: {total_norm:.4f}")

                    model_optim.step()

                # current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                # max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # print("调用self.vali")
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if np.isnan(vali_loss).any():
                print("Warning: Validation loss is NaN!")
            test_loss = self.vali(test_data, test_loader, criterion)

            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            self.logger.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                # print("Early stopping")
                self.logger.info("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                # print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
                self.logger.info('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        self.model.mode_now = "test"
        test_data, test_loader = self._get_data(flag='test')

        if test:
            # print('loading model')
            self.logger.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(test_loader):
                # 1. 数据迁移至设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # 2. 构建 Decoder 输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 💡 3. 构建全能参数字典
                model_kwargs = {
                    "x_enc": batch_x,
                    "x_mark_enc": batch_x_mark,
                    "x_dec": dec_inp,
                    "x_mark_dec": batch_y_mark,
                    "batch_y": batch_y, 
                    "cycle_index": batch_cycle,
                }


                # 💡 4. 一键处理 AMP 和 前向传播
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(**model_kwargs)
                    
                    # 统一处理返回值：不管返回的是 (output, loss) 还是 (output, attention)
                    # 我们在 test 阶段只关心第一个预测结果
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        outputs = outputs[0]

                # 5. 截取特征维度与预测长度
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                # 6. 转换回 Numpy 并记录
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                # inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)

                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    # np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                    # np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.concatenate(inputx, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        ### denorm ###
        # denorm_preds = np.stack([test_data.inverse_transform(pred) for pred in preds])
        # denorm_trues = np.stack([test_data.inverse_transform(true) for true in trues])

        ### denorm ###

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        # mae, mse, rmse, mape, mspe, rse, corr = metric(denorm_preds, denorm_trues)

        # print('mse:{}, mae:{}'.format(mse, mae))
        self.logger.info('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
                # 1. 数据统一迁移至设备 (修复了原代码 batch_y 未统一 to device 的问题)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                batch_cycle = batch_cycle.int().to(self.device)

                # 2. 构建 Decoder 输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 💡 3. 构建全能参数字典
                model_kwargs = {
                    "x_enc": batch_x,
                    "x_mark_enc": batch_x_mark,
                    "x_dec": dec_inp,
                    "x_mark_dec": batch_y_mark,
                    "batch_y": batch_y,
                    "cycle_index": batch_cycle,
                }


                # 💡 4. 一键处理 AMP 和 前向传播
                with torch.cuda.amp.autocast(enabled=self.args.use_amp):
                    outputs = self.model(**model_kwargs)
                    
                    # 统一处理返回值：预测阶段我们只关心预测矩阵本身
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        outputs = outputs[0]

                # 5. 转换回 Numpy 并记录预测结果
                pred = outputs.detach().cpu().numpy()  # .squeeze() 如果你需要的话
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
