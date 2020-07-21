import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import os
import tqdm
import pylab as plt
import numpy as np
import scipy.sparse as sps
from collections.abc import Sequence
import time
from src import utils as ut
from sklearn.metrics import confusion_matrix
import skimage
from haven import haven_utils as hu
from haven import haven_img as hi
from torchvision import transforms
from src import models
from src.models import base_networks

from skimage.segmentation import mark_boundaries
from skimage.color import label2rgb

from . import losses, metrics_for_onnx as metrics

import onnx
import onnxruntime
import cv2
from google.colab.patches import cv2_imshow
from onnx import optimizer as onnx_optimizer
                    
class SemSeg(torch.nn.Module):
    def __init__(self, exp_dict, train_set):
        super().__init__()
        self.exp_dict = exp_dict
        self.n_classes = train_set.n_classes
        self.exp_dict = exp_dict

        self.model_base = models.base_networks.get_base(self.exp_dict['model'].get('base', 'unet2d'),
                                               self.exp_dict, n_classes=self.n_classes)

        if self.exp_dict["optimizer"] == "adam":
            self.opt = torch.optim.Adam(
                self.model_base.parameters(), lr=self.exp_dict["lr"], betas=(0.99, 0.999))

        elif self.exp_dict["optimizer"] == "sgd":
            self.opt = torch.optim.SGD(
                self.model_base.parameters(), lr=self.exp_dict["lr"])

        else:
            raise ValueError

        self.just_one = True
        self.sess = None
        
    def get_state_dict(self):
        state_dict = {"model": self.model_base.state_dict(),
                      "opt": self.opt.state_dict()}

        return state_dict

    def load_state_dict(self, state_dict):
        self.model_base.load_state_dict(state_dict["model"])
        if 'opt' not in state_dict:
            return
        self.opt.load_state_dict(state_dict["opt"])

    # def train_on_loader(self, train_loader):
    #     self.train()

    #     n_batches = len(train_loader)

    #     pbar = tqdm.tqdm(desc="Training", total=n_batches, leave=False)
    #     train_monitor = TrainMonitor()
    
    #     for batch in train_loader:
    #         score_dict = self.train_on_batch(batch)
            
    #         train_monitor.add(score_dict)
    #         msg = ' '.join(["%s: %.3f" % (k, v) for k,v in train_monitor.get_avg_score().items()])
    #         pbar.set_description('Training - %s' % msg)
    #         pbar.update(1)
            
    #     pbar.close()

    #     return train_monitor.get_avg_score()

    @torch.no_grad()
    def val_on_loader(self, val_loader, savedir_images=None, n_images=0, save_preds=False):
        self.eval()

        # metrics
        seg_monitor = metrics.SegMonitor()
        n_batches = len(val_loader)
        pbar = tqdm.tqdm(desc="Validating", total=n_batches, leave=False)
        
        for i, batch in enumerate(val_loader):
            if self.just_one:
                net_name = self.exp_dict['model'].get('base', 'unet2d')
                #print(net_name)
                if net_name == 'unet2d':
                    torch.onnx._export(self.model_base, batch['images'].cuda(), 'onnx_model_unet2d.onnx', verbose=False, opset_version=12, keep_initializers_as_inputs=True, export_params=True)
                    print('Exported to onnx_model_unet2d.onnx')
                    onnx_model = onnx.load("onnx_model_unet2d.onnx")
                    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
                    optimized_model = onnx_optimizer.optimize(onnx_model, passes)
                    onnx.save(optimized_model, "optimized_model_unet2d.onnx")
                    content = optimized_model.SerializeToString()
                    self.sess = onnxruntime.InferenceSession(content)
                elif net_name == 'pspnet':
                    self.onnx_model = torch.onnx._export(self.model_base, batch['images'].cuda(), 'onnx_model_pspnet.onnx', verbose=False, opset_version=12, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, keep_initializers_as_inputs=True, export_params=True)
                    print('Exported to onnx_model_pspnet.onnx')
                    onnx_model = onnx.load("onnx_model_pspnet.onnx")
                    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
                    optimized_model = onnx_optimizer.optimize(onnx_model, passes)
                    onnx.save(optimized_model, "optimized_model_pspnet.onnx")
                    content = optimized_model.SerializeToString()
                    self.sess = onnxruntime.InferenceSession(content)
                else:
                    print('not supported network to export.')
                self.just_one = False
            seg_monitor.val_on_batch(self, batch)

            #print(batch['images'].numpy().shape)
            #x = self.predict_on_batch_onnx(batch)#, onnx_model)
            #x = self.predict_on_batch_onnx(batch).cpu().numpy()
            #y = self.predict_on_batch(batch).to(torch.device("cpu")).numpy()
            #y = self.predict_on_batch(batch).cpu().numpy()
            #print(x, np.argmax(x), y, np.argmax(y))
            #print(torch.from_numpy(x).shape)
            # cv2.imwrite('Save/onnx_pred_{}.jpg'.format(i), x[0]/np.max(x))
            # cv2.imwrite('Save/torch_pred_{}.jpg'.format(i), y[0]/np.max(y))
            # cv2_imshow(x[0])
            # cv2_imshow(y[0])
            pbar.update(1)

            if savedir_images and i < n_images:
                os.makedirs(savedir_images, exist_ok=True)
                self.vis_on_batch(batch, savedir_image=os.path.join(
                   savedir_images, "%d.jpg" % i), save_preds=save_preds)
                pbar.set_description("Validating & Saving Images: %.4f mIoU" %
                                 (seg_monitor.get_avg_score()['val_score']))
            else:
                pbar.set_description("Validating: %.4f mIoU" %
                                 (seg_monitor.get_avg_score()['val_score']))
# remove break to scan all images and get the full score
            #break

        pbar.close()
        val_dict = seg_monitor.get_avg_score()
        out_dict = {}
        for c in range(self.n_classes):
            out_dict['iou_group%d' % c] = val_dict['iou'][c]
        out_dict['val_score'] = val_dict['val_score']
        return out_dict

    # def train_on_batch(self, batch, **extras):
    #     self.train()
    #     self.model_base.train()
    #     self.opt.zero_grad()

    #     images, labels = batch["images"], batch["masks"]
    #     images, labels = images.cuda(), labels.cuda()
        
    #     logits = self.model_base(images)
    #     # match image size
    #     logits = match_image_size(images, logits)
    #     # compute loss
    #     loss_name = self.exp_dict['model'].get('loss', 'cross_entropy')
    #     loss = losses.compute_loss(loss_name, logits, labels)
        
    #     if loss != 0:
    #         loss.backward()

    #     self.opt.step()

    #     return {"train_loss": float(loss)}

    def predict_on_batch(self, batch):
        #print('image shape before cuda', batch["images"].shape)
        images = batch["images"].cuda()
        #print(images = batch["images"].cuda().shape)
        n = images.shape[0]
        logits = self.model_base.forward(images)
        # match image size
        logits = match_image_size(images, logits)
        #print('main logits shape', logits.shape)
        return logits.argmax(dim=1)

    def predict_on_batch_onnx(self, batch):#, onnx_model):
        images = batch['images'].numpy()
        #print('image shape', images.shape)
        n = images.shape[0]
        ## onnx
        #content = self.onnx_model.SerializeToString()
        #sess = onnxruntime.InferenceSession(content)
        sess = self.sess
        feed = dict([(input.name, images) for n, input in enumerate(sess.get_inputs())])
        logits = np.asarray(sess.run(None, feed))
        logits = np.squeeze(logits, axis=0)
        #print('logits shape', logits.shape)
        # match image size
        logits = match_image_size(images, logits)
        return torch.from_numpy(np.argmax(logits, axis=1))

    @torch.no_grad()
    def vis_on_batch(self, batch, savedir_image, save_preds=False):
        # os.makedirs(savedir_image, exist_ok=True)
        self.eval()

        # if self.just_one:
        #   onnx_output = torch.onnx._export(self.model_base, batch['images'].cuda(), 'onnx_model.onnx', verbose=False, opset_version=12)
        #   self.just_one = False

        # onnx_output = torch.onnx._export(self.model_base, torch_input.cuda(), 'onnx_model2.onnx', verbose=False, opset_version=11, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)


        # clf
        #pred_mask = self.predict_on_batch(batch).cpu()
        pred_mask = self.predict_on_batch_onnx(batch).cpu()

        # print(pred_mask.sum())
        img = hu.f2l(batch['images'])[0]
        img += abs(img.min())
        img /= img.max()
        img = img.repeat(1,1,3)

        mask_vis = batch["masks"].clone().float()[0][..., None]
        mask_vis[mask_vis == 255] = 0

        pred_mask_vis = pred_mask.clone().float()[0][..., None]
        vmax = 0.1

        fig, ax_list = plt.subplots(ncols=3, nrows=1)
        ax_list[0].imshow(img[:, :, 0], cmap='gray',
                        #   interpolation='sinc', vmin=0, vmax=0.4
                          )

        colors_all = np.array(['black', 'red', 'blue', 'green', 'purple'])
        colors = colors_all[np.unique(mask_vis).astype(int)]

        vis = label2rgb(mask_vis[:, :, 0].numpy(), image=img.numpy(
        ), colors=colors, bg_label=255, bg_color=None, alpha=0.6, kind='overlay')
        vis = mark_boundaries(
            vis, mask_vis[:, :, 0].numpy().astype('uint8'), color=(1, 1, 1))

        ax_list[1].imshow(vis, cmap='gray')

        colors = colors_all[np.unique(pred_mask_vis).astype(int)]
        vis = label2rgb(pred_mask_vis[:, :, 0].numpy(), image=img.numpy(
        ), colors=colors, bg_label=255, bg_color=None, alpha=0.6, kind='overlay')
        vis = mark_boundaries(
            vis, pred_mask_vis[:, :, 0].numpy().astype('uint8'), color=(1, 1, 1))

        ax_list[2].imshow(vis, cmap='gray')

        for i in range(1, self.n_classes):
            plt.plot([None], [None], label='group %d' % i, color=colors_all[i])
        # ax_list[1].axis('off')
        ax_list[0].grid()
        ax_list[1].grid()
        ax_list[2].grid()

        ax_list[0].tick_params(axis='x', labelsize=6)
        ax_list[0].tick_params(axis='y', labelsize=6)

        ax_list[1].tick_params(axis='x', labelsize=6)
        ax_list[1].tick_params(axis='y', labelsize=6)

        ax_list[2].tick_params(axis='x', labelsize=6)
        ax_list[2].tick_params(axis='y', labelsize=6)

        ax_list[0].set_title('Original image', fontsize=8)
        ax_list[1].set_title('Ground-truth',  fontsize=8)
        ax_list[2].set_title('Prediction',  fontsize=8)

        legend_kwargs = {"loc": 2, "bbox_to_anchor": (1.05, 1),
                         'borderaxespad': 0., "ncol": 1}
        ax_list[2].legend(fontsize=6, **legend_kwargs)
        plt.savefig(savedir_image.replace('.jpg', '.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

        # save predictions
        if save_preds:
            from PIL import Image
            pred_dict = {}
            pred_numpy = pred_mask.cpu().numpy().squeeze().astype('uint8')

            uniques = np.unique(np.array(pred_numpy))
            # print(uniques)
            meta_dict = batch['meta'][0]

            for u in range(self.n_classes):
                meta_dict['gt_group%d_n_pixels'%u] = float((batch['masks']==u).float().sum())
                meta_dict['pred_group%d_n_pixels'%u] = float((pred_mask==u).float().sum())
                
                if u == 0:
                    continue
                pred = Image.fromarray(pred_numpy==u)
                pred.save(savedir_image.replace('.jpg', '_group%d.png'%u))

            hu.save_json(savedir_image.replace('.jpg', '.json'), meta_dict)
            

def match_image_size(images, logits):
    h, w = images.shape[-2:]
    hl, wl = logits.shape[-2:]
    if hl != h or wl != w:
        logits = F.interpolate(logits, (h,w), mode='bilinear', align_corners=True)
    return logits

# class TrainMonitor:
#     def __init__(self):
#         self.score_dict_sum = {}
#         self.n = 0

#     def add(self, score_dict):
#         for k,v in score_dict.items():
#             if k not in self.score_dict_sum:
#                 self.score_dict_sum[k] = score_dict[k]
#             else:
#                 self.n += 1
#                 self.score_dict_sum[k] += score_dict[k]

#     def get_avg_score(self):
#         return {k:v/(self.n + 1) for k,v in self.score_dict_sum.items()}


