import time
from enum import Enum
from functools import reduce

import numpy as np
import sparseconvnet as scn
import torch
from torch import nn
from torch.nn import functional as F

import torchplus
from torchplus import metrics
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import (WeightedSigmoidClassificationLoss,
                                          WeightedSmoothL1LocalizationLoss,
                                          WeightedSoftmaxClassificationLoss)
from second.pytorch.models.pointpillars import PillarFeatureNet, PointPillarsScatter
from second.pytorch.models.fvfeature import FrontViewFeatureNet, FrontViewScatter
from second.pytorch.utils import get_paddings_indicator

from second.pytorch.models.dlav0 import get_pose_net
from second.pytorch.models.dla_dcn import get_pose_net as get_pose_net_dcn
from second.pytorch.models.losses import FocalLoss, L1Loss, BinRotLoss
from second.pytorch.models.decode import ddd_decode
from second.utils.post_process import ddd_post_process
from second.utils.debugger import Debugger

import cv2

def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, name='vfe'):
        super(VFELayer, self).__init__()
        self.name = name
        self.units = int(out_channels / 2)
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]

        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        # [K, 1, units]
        repeated = aggregated.repeat(1, voxel_count, 1)

        concatenated = torch.cat([pointwise, repeated], dim=2)
        # [K, T, 2 * units]
        return concatenated


class VoxelFeatureExtractor(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) == 2
        num_input_features += 3  # add mean features
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        self.vfe1 = VFELayer(num_input_features, num_filters[0], use_norm)
        self.vfe2 = VFELayer(num_filters[0], num_filters[1], use_norm)
        self.linear = Linear(num_filters[1], num_filters[1])
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = BatchNorm1d(num_filters[1])

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat(
                [features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        # mask = features.max(dim=2, keepdim=True)[0] != 0
        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)
        x *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(x, dim=1)[0]
        return voxelwise


class VoxelFeatureExtractorV2(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractorV2, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) > 0
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        num_filters = [num_input_features] + num_filters
        filters_pairs = [[num_filters[i], num_filters[i + 1]]
                         for i in range(len(num_filters) - 1)]
        self.vfe_layers = nn.ModuleList(
            [VFELayer(i, o, use_norm) for i, o in filters_pairs])
        self.linear = Linear(num_filters[-1], num_filters[-1])
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = BatchNorm1d(num_filters[-1])

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat(
                [features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        for vfe in self.vfe_layers:
            features = vfe(features)
            features *= mask
        features = self.linear(features)
        features = self.norm(features.permute(0, 2, 1).contiguous()).permute(
            0, 2, 1).contiguous()
        features = F.relu(features)
        features *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(features, dim=1)[0]
        return voxelwise


class SparseMiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SparseMiddleExtractor'):
        super(SparseMiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape
        middle_layers = []

        num_filters = [num_input_features] + num_filters_down1
        # num_filters = [64] + num_filters_down1
        filters_pairs_d1 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]

        for i, o in filters_pairs_d1:
            middle_layers.append(scn.SubmanifoldConvolution(3, i, o, 3, False))
            middle_layers.append(scn.BatchNormReLU(o, eps=1e-3, momentum=0.99))
        middle_layers.append(
            scn.Convolution(
                3,
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        middle_layers.append(
            scn.BatchNormReLU(num_filters[-1], eps=1e-3, momentum=0.99))
        # assert len(num_filters_down2) > 0
        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2
        filters_pairs_d2 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]
        for i, o in filters_pairs_d2:
            middle_layers.append(scn.SubmanifoldConvolution(3, i, o, 3, False))
            middle_layers.append(scn.BatchNormReLU(o, eps=1e-3, momentum=0.99))
        middle_layers.append(
            scn.Convolution(
                3,
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        middle_layers.append(
            scn.BatchNormReLU(num_filters[-1], eps=1e-3, momentum=0.99))
        middle_layers.append(scn.SparseToDense(3, num_filters[-1]))
        self.middle_conv = Sequential(*middle_layers)

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()[:, [1, 2, 3, 0]]
        ret = self.scn_input((coors.cpu(), voxel_features, batch_size))
        ret = self.middle_conv(ret)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class ZeroPad3d(nn.ConstantPad3d):
    def __init__(self, padding):
        super(ZeroPad3d, self).__init__(padding, 0)


class MiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='MiddleExtractor'):
        super(MiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm3d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm3d)
            # BatchNorm3d = change_default_args(
            #     group=32, eps=1e-3, momentum=0.01)(GroupBatchNorm3d)
            Conv3d = change_default_args(bias=False)(nn.Conv3d)
        else:
            BatchNorm3d = Empty
            Conv3d = change_default_args(bias=True)(nn.Conv3d)
        self.voxel_output_shape = output_shape
        self.middle_conv = Sequential(
            ZeroPad3d(1),
            Conv3d(num_input_features, 64, 3, stride=(2, 1, 1)),
            BatchNorm3d(64),
            nn.ReLU(),
            ZeroPad3d([1, 1, 1, 1, 0, 0]),
            Conv3d(64, 64, 3, stride=1),
            BatchNorm3d(64),
            nn.ReLU(),
            ZeroPad3d(1),
            Conv3d(64, 64, 3, stride=(2, 1, 1)),
            BatchNorm3d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        output_shape = [batch_size] + self.voxel_output_shape[1:]
        ret = scatter_nd(coors.long(), voxel_features, output_shape)
        # print('scatter_nd fw:', time.time() - t)
        ret = ret.permute(0, 4, 1, 2, 3)
        ret = self.middle_conv(ret)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret

'''
class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_filters=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 name='rpn'):
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        if use_bev:
            self.bev_extractor = Sequential(
                Conv2d(6, 32, 3, padding=1),
                BatchNorm2d(32),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),
                Conv2d(32, 64, 3, padding=1),
                BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            block2_input_filters += 64

        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_filters, num_filters[0], 3, stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * 2, 1)

    def forward(self, x, bev=None):
        x = self.block1(x)
        up1 = self.deconv1(x)
        if self._use_bev:
            bev[:, -1] = torch.clamp(
                torch.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
            x = torch.cat([x, self.bev_extractor(bev)], dim=1)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        return ret_dict

'''
class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


class VoxelNet(nn.Module):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 name='voxelnet',
                 save_path=None,
                 RGB_embedding=False,
                 occupancy_embedding=False):

        super().__init__()
        self.name = name
        self._num_class = num_class
        self._encode_background_as_zeros = encode_background_as_zeros
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0

        self._save_path = save_path
        self._RGB_embedding = RGB_embedding
        self._occupancy_embedding = occupancy_embedding

        self._output_shape = output_shape

        self.hm_weight, self.dep_weight, self.rot_weight, self.dim_weight, self.off_weight = 1, 1, 1, 1, 1
        self.crit = FocalLoss()
        self.crit_reg = L1Loss()
        self.crit_rot = BinRotLoss()

        input_channel = 5
        if self._RGB_embedding:
            input_channel += 3
        if self._occupancy_embedding:
            input_channel += 1

        heads = {'hm':num_class, 'dep':1, 'rot':8, 'dim':3, 'reg':2}
        self._down_ratio = 1
        # self.rpn = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=self._down_ratio, input_channel=input_channel)
        self.rpn = get_pose_net_dcn(num_layers=34, heads=heads, head_conv=256, down_ratio=self._down_ratio, input_channel=input_channel)

        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=use_sigmoid_score,
            encode_background_as_zeros=encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", torch.LongTensor(1).zero_())

        self.debug_mode = True
        self.save_imgs = False
        if self.debug_mode:
            self.debugger = Debugger(theme='black', num_classes=self._num_class, down_ratio=self._down_ratio)

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def get_fvmap(self, voxels, coords, batch_size, RGB_embedding=False):
        # voxels = torch.mean(voxels, dim=1, keepdim=True)
        voxels = voxels[:, 0, :]
        voxel_features = voxels.squeeze()
        if not RGB_embedding:
            channel_idx = 3
            nx, ny, nchannels = self._output_shape[3], self._output_shape[2], 1
        else:
            nx, ny, nchannels = self._output_shape[3], self._output_shape[2], 3
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(nchannels, nx * ny, dtype=voxel_features.dtype,
                                 device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            if not RGB_embedding:
                canvas[:, indices] = voxels[channel_idx, :]
            else:
                canvas[:, indices] = voxels[5:, :]

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, nchannels, ny, nx)

        spherical_points = batch_canvas.cpu().numpy()
        return spherical_points

    def make_input_img(self, fv_image, batch_size, batch_imgidx, RGB_embedding=False, channel_idx=3, id_suffix='D'):
        for i in range(batch_size):
            fvmap = fv_image.detach().cpu().numpy()[i]
            if not RGB_embedding:
                fvmap = fvmap[channel_idx:channel_idx+1, :, :]
                mask = fvmap != 0
                # fvmap[mask] -= fvmap[mask].min()
                # fvmap[mask] /= fvmap[mask].max()
                # fvmap[mask] = 1 - fvmap[mask]

                text_list = []
                text_list.append(id_suffix)
                text_list.append('min: ' + str(fvmap[mask].min()))
                text_list.append('max: ' + str(fvmap[mask].max()))

                colormap = self.debugger.gen_colormap(fvmap, text_list=text_list)
                self.debugger.add_img(colormap, img_id=str(batch_imgidx[i]) + '_' + id_suffix)
            else:
                colormap = self.debugger.gen_colormap_RGB(fvmap[channel_idx:channel_idx+3, :, :])
                self.debugger.add_img(colormap, img_id=str(batch_imgidx[i]) + '_' + id_suffix)

                # filtered_colormap = self.debugger.get_filtered_image(colormap)
                # for idx, filtered_img in enumerate(filtered_colormap):
                #     self.debugger.add_img(filtered_img, img_id=str(batch_imgidx[i]) + '_' + id_suffix + '_' + str(idx))


    def make_channel_img(self, hm, batch_size, batch_imgidx, id):
        for i in range(batch_size):
            heatmap = hm[i]
            colormap = self.debugger.gen_colormap(heatmap)
            self.debugger.add_img(colormap, img_id=str(batch_imgidx[i]) + '_' + id)



    def forward(self, example):
        """module's forward should always accept dict and return loss.
        """
        # forward_time = time.time()
        # print("@@@@@@@@@@@@@@@ Forward")

        batch_size_dev = example['image_idx'].shape[0]
        t = time.time()

        fv_image = example['fv_image']
        preds_dict = self.rpn(fv_image)
        self._total_forward_time += time.time() - t
        if self.training:
            if self.debug_mode:
                self.make_input_img(example['fv_image'], batch_size_dev, example['image_idx'], channel_idx=0, id_suffix='X')
                self.make_input_img(example['fv_image'], batch_size_dev, example['image_idx'], channel_idx=1, id_suffix='Y')
                self.make_input_img(example['fv_image'], batch_size_dev, example['image_idx'], channel_idx=2, id_suffix='Z')
                self.make_input_img(example['fv_image'], batch_size_dev, example['image_idx'], channel_idx=3, id_suffix='D')
                self.make_input_img(example['fv_image'], batch_size_dev, example['image_idx'], channel_idx=4, id_suffix='R')
                if self._occupancy_embedding:
                    occupancy_idx = 5
                    if self._RGB_embedding:
                        occupancy_idx += 3
                    self.make_input_img(example['fv_image'], batch_size_dev, example['image_idx'], channel_idx=occupancy_idx, id_suffix='O')


                if self._RGB_embedding:
                    self.make_input_img(example['fv_image'], batch_size_dev, example['image_idx'], RGB_embedding=True,
                                        channel_idx=5, id_suffix='RGB')

                    for i in range(batch_size_dev):
                        RGB_image = example["RGB_image"][i]
                        # RGB_image = RGB_image * 255.0
                        self.debugger.add_img(RGB_image, img_id='RGB '+ str(example['image_idx'][i]))

                spherical_gt_boxes = example['spherical_gt_boxes']
                for i in range(spherical_gt_boxes.shape[0]):
                    num_obj = 0
                    for j in range(spherical_gt_boxes.shape[1]):
                        if np.all(spherical_gt_boxes[i, j, :] == 0):
                            num_obj = j
                            break
                    if num_obj == 0:
                        continue
                    spherical_gt_box = torch.from_numpy(spherical_gt_boxes[i, :num_obj, :])

                    phi = spherical_gt_box[:, 0] * example['grid_size'][i][0] + example['meta'][i]['phi_min']
                    theta = spherical_gt_box[:, 1] * example['grid_size'][i][1] + example['meta'][i]['theta_min']
                    depth = spherical_gt_box[:, 2]
                    spherical_gt_box[:, 0] = torch.sin(theta) * torch.cos(phi) * depth
                    spherical_gt_box[:, 1] = torch.sin(theta) * torch.sin(phi) * depth
                    spherical_gt_box[:, 2] = (torch.cos(theta) * depth) - (spherical_gt_box[:, 5] / 2)

                    locs = spherical_gt_box[:, 0:3]
                    dims = spherical_gt_box[:, 3:6]
                    angles = spherical_gt_box[:, 6]
                    camera_box_origin = [0.5, 0.5, 0]

                    box_corners = box_torch_ops.center_to_corner_box3d(
                        locs, dims, angles, camera_box_origin, axis=2)
                    box_corners_in_image = box_torch_ops.project_to_fv_image(
                        box_corners, example['grid_size'][i], example['meta'][i])
                    box_centers_in_image = box_torch_ops.project_to_fv_image(
                        locs, example['grid_size'][i], example['meta'][i])

                    for j in range(num_obj):
                        self.debugger.add_3d_detection2(box_corners_in_image[j], c=[0, 255, 0],
                                                        img_id=str(example['image_idx'][i])+'_R')
                        self.debugger.add_point(box_centers_in_image[j], c=(0, 255, 0), img_id=str(example['image_idx'][i])+'_R')


                if self.save_imgs:
                    self.debugger.save_all_imgs(path=str(self._save_path) + '/visualize/', stack=True, ids=example['image_idx'])
                else:
                    self.debugger.show_all_imgs(pause=True, stack=True, ids=example['image_idx'])
                self.debugger.remove_all_imgs()

            # labels = example['labels']
            output = preds_dict

            output['hm'] = torch.clamp(output['hm'].sigmoid_(), min=1e-4, max=1 - 1e-4)
            output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
            
            label_hm = torch.from_numpy(example['hm']).cuda()
            label_dep = torch.from_numpy(example['dep']).cuda()
            label_dim = torch.from_numpy(example['dim']).cuda()
            label_rotbin = torch.from_numpy(example['rotbin']).cuda()
            label_rotres = torch.from_numpy(example['rotres']).cuda()
            label_reg = torch.from_numpy(example['reg']).cuda()
            label_ind = torch.from_numpy(example['ind']).cuda()
            label_reg_mask = torch.from_numpy(example['reg_mask']).cuda()
            label_rot_mask = torch.from_numpy(example['rot_mask']).cuda()
            # label_fg_mask = torch.from_numpy(example['fg_mask']).cuda()

            hm_loss = self.crit(output['hm'], label_hm)
            # dep_loss = self.crit_reg(output['dep'], label_fg_mask, label_dep, div=True)
            # dim_loss = self.crit_reg(output['dim'], label_fg_mask, label_dim)
            # rot_loss = self.crit_rot(output['rot'], label_fg_mask, label_rotbin, label_rotres)
            # off_loss = self.crit_reg(output['reg'], label_fg_mask, label_reg)
            dep_loss = self.crit_reg(output['dep'], label_reg_mask, label_ind, label_dep)
            dim_loss = self.crit_reg(output['dim'], label_reg_mask, label_ind, label_dim)
            rot_loss = self.crit_rot(output['rot'], label_rot_mask, label_ind, label_rotbin, label_rotres)
            off_loss = self.crit_reg(output['reg'], label_reg_mask, label_ind, label_reg)
            loss = self.hm_weight * hm_loss + self.dep_weight * dep_loss + \
                   self.dim_weight * dim_loss + self.rot_weight * rot_loss + \
                   self.off_weight * off_loss
            # t_total = time.time() - forward_time
            # print("t_total: ", t_total)     # 0.32
            return {'loss': loss, 'hm_loss': hm_loss, 'dep_loss': dep_loss,
                          'dim_loss': dim_loss, 'rot_loss': rot_loss, 'off_loss': off_loss}

        else:
            return self.predict(example, preds_dict)

    def predict(self, example, preds_dict):
        # print("########################## Predict")
        t = time.time()
        # batch_size = example['anchors'].shape[0]
        # batch_anchors = example["anchors"].view(batch_size, -1, 7)
        batch_size = example['image_idx'].shape[0]

        self._total_inference_count += batch_size
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        batch_imgidx = example['image_idx']
        # print("image_idx: ", batch_imgidx[0])
        # print('image_path: ' + example['image_path'])

        self._total_forward_time += time.time() - t
        t = time.time()

        # torch.cuda.synchronize()
        output = preds_dict
        output['hm'] = output['hm'].sigmoid_()
        output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
        # wh = output['wh'] if self.opt.reg_bbox else None
        reg = output['reg']
        # torch.cuda.synchronize()

        score_thresh = 0.1

        dets = ddd_decode(output['hm'], output['rot'], output['dep'],
                          output['dim'], reg=reg, K=40)
        if self.debug_mode:
            # self.make_input_img(example['fv_image'], batch_size, example['image_idx'], channel_idx=0, id_suffix='X')
            # self.make_input_img(example['fv_image'], batch_size, example['image_idx'], channel_idx=1, id_suffix='Y')
            # self.make_input_img(example['fv_image'], batch_size, example['image_idx'], channel_idx=2, id_suffix='Z')
            self.make_input_img(example['fv_image'], batch_size, example['image_idx'], channel_idx=3, id_suffix='D')
            self.make_input_img(example['fv_image'], batch_size, example['image_idx'], channel_idx=4, id_suffix='R')
            # self.make_channel_img(output['hm'].detach().cpu().numpy(), batch_size, example['image_idx'], id='HM')

            # for i in range(batch_size):
            #     pos_mask = dets[i, :, 2] > score_thresh
            #     self.debugger.add_points(dets[i,:,:2][pos_mask].view(1, -1, 2).detach().cpu().numpy().astype(np.int32), img_id='input_'+str(i))

            if self._RGB_embedding:
                self.make_input_img(example['fv_image'], batch_size, example['image_idx'], RGB_embedding=True,
                                    channel_idx=5, id_suffix='RGB')
                for i in range(batch_size):
                    RGB_image = example["RGB_image"][i]
                    self.debugger.add_img(RGB_image, img_id='RGB ' + str(example['image_idx'][i]))

        dets = self.post_process(dets, example['meta'], example['grid_size'])

        predictions_dicts = []

        selected_boxes = None
        for i in range(len(dets)):
            for cls_id, det in dets[i].items():
                det = torch.from_numpy(det).float().cuda()
                selected_boxes = torch.nonzero(det[:, 8] >= score_thresh).cuda()
                # selected_idx = selected_boxes[0]
                img_idx = batch_imgidx[i]

                if selected_boxes.shape[0] > 0:
                # if len(selected_idx) > 0:
                    # alpha = det[selected_boxes, 0]
                    # dim = det[selected_boxes, 1:4]
                    # loc = det[selected_boxes, 4:7]
                    # ry = det[selected_boxes, 7]
                    # score = det[selected_boxes, 8]
                    selected_idx = selected_boxes[:, 0]
                    label_preds = torch.ones((selected_idx.shape[0], 1)).cuda() * cls_id
                    final_scores = det[selected_idx, 8]
                    final_box_preds_lidar = det[selected_idx, 1:8]
                    final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                        final_box_preds_lidar, batch_rect[i], batch_Trv2c[i])

                    # print(final_box_preds_camera)
                    # final_box_preds_camera[:,:3] = 0
                    locs = final_box_preds_camera[:, :3]
                    dims = final_box_preds_camera[:, 3:6]
                    angles = final_box_preds_camera[:, 6]
                    camera_box_origin = [0.5, 1.0, 0.5]
                    box_corners = box_torch_ops.center_to_corner_box3d(
                        locs, dims, angles, camera_box_origin, axis=1)
                    box_corners_in_image = box_torch_ops.project_to_image(
                        box_corners, batch_P2[i])

                    # box_corners_in_image: [N, 8, 2]
                    minxy = torch.min(box_corners_in_image, dim=1)[0]
                    maxxy = torch.max(box_corners_in_image, dim=1)[0]


                    locs_fv = final_box_preds_lidar[:, 3:6]
                    dims_fv = final_box_preds_lidar[:, 0:3]
                    angles_fv = final_box_preds_lidar[:, 6]
                    camera_box_origin_fv = [0.5, 0.5, 0]
                    box_corners_fv = box_torch_ops.center_to_corner_box3d(
                        locs_fv, dims_fv, angles_fv, camera_box_origin_fv, axis=2)

                    box_corners_in_fv_image = box_torch_ops.project_to_fv_image(
                        box_corners_fv, example['grid_size'][i], example['meta'][i])
                    # box_centers_in_image = box_torch_ops.project_to_fv_image(
                    #     locs, example['voxel_size'][i], example['meta'][i])



                    # resized_image_shape = example['resized_image_shape'][i]
                    # resized_w, resized_h = resized_image_shape[0], resized_image_shape[1]
                    # image_shape = example['image_shape'][i]
                    # original_w, original_h = image_shape[1], image_shape[0]
                    # scale_w = original_w / resized_w
                    # scale_h = original_h / resized_h
                    # if scale_w != 1:
                    #     minxy[:, 0] = minxy[:, 0] * scale_w
                    #     maxxy[:, 0] = maxxy[:, 0] * scale_w
                    # if scale_h != 1:
                    #     minxy[:, 1] = minxy[:, 1] * scale_h
                    #     maxxy[:, 1] = maxxy[:, 1] * scale_h

                    box_2d_preds = torch.cat([minxy, maxxy], dim=1)

                    # print(final_box_preds_camera)
                    # predictions
                    predictions_dict = {
                        "bbox": box_2d_preds,
                        "box3d_camera": final_box_preds_camera,
                        "box3d_lidar": final_box_preds_lidar,
                        "scores": final_scores,
                        "label_preds": label_preds,
                        "image_idx": img_idx,
                        "box_corners_in_image": box_corners_in_image,
                        "box_corners_in_fv_image": box_corners_in_fv_image,
                        # "box_centers_in_image": box_centers_in_image
                    }
                else:
                    predictions_dict = {
                        "bbox": None,
                        "box3d_camera": None,
                        "box3d_lidar": None,
                        "scores": None,
                        "label_preds": None,
                        "image_idx": img_idx,
                        "box_corners_in_image": None,
                        "box_corners_in_fv_image": None,
                        # "box_centers_in_image": None
                    }
                predictions_dicts.append(predictions_dict)
        self._total_postprocess_time += time.time() - t

        if self.debug_mode:
            spherical_gt_boxes = example['spherical_gt_boxes']
            for i in range(spherical_gt_boxes.shape[0]):
                num_obj = 0
                for j in range(spherical_gt_boxes.shape[1]):
                    if np.all(spherical_gt_boxes[i, j, :] == 0):
                        num_obj = j
                        break
                if num_obj == 0:
                    continue
                spherical_gt_box = torch.from_numpy(spherical_gt_boxes[i, :num_obj, :])

                phi = spherical_gt_box[:, 0] * example['grid_size'][i][0] + example['meta'][i]['phi_min']
                theta = spherical_gt_box[:, 1] * example['grid_size'][i][1] + example['meta'][i]['theta_min']
                depth = spherical_gt_box[:, 2]
                spherical_gt_box[:, 0] = torch.sin(theta) * torch.cos(phi) * depth
                spherical_gt_box[:, 1] = torch.sin(theta) * torch.sin(phi) * depth
                spherical_gt_box[:, 2] = (torch.cos(theta) * depth) - (spherical_gt_box[:, 5] / 2)

                locs = spherical_gt_box[:, 0:3]
                dims = spherical_gt_box[:, 3:6]
                angles = spherical_gt_box[:, 6]
                camera_box_origin = [0.5, 0.5, 0]

                box_corners = box_torch_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=2)
                box_corners_in_fv_image = box_torch_ops.project_to_fv_image(
                    box_corners, example['grid_size'][i], example['meta'][i])
                # box_centers_in_image = box_torch_ops.project_to_fv_image(
                #     locs, example['voxel_size'][i], example['meta'][i])

                for j in range(num_obj):
                    # self.debugger.add_3d_detection2(box_corners_in_fv_image[j], c= [0,255, 0], img_id=str(batch_imgidx[i])+'_D')
                    # self.debugger.add_point(box_centers_in_image[j], c= (0, 255, 0), img_id=str(batch_imgidx[i])+'_D')
                    self.debugger.add_3d_detection2(box_corners_in_fv_image[j], c=[0, 255, 0],
                                                    img_id=str(batch_imgidx[i]) + '_D')
                    # self.debugger.add_point(box_centers_in_image[j], c=(0, 255, 0), img_id=str(batch_imgidx[i]) + '_RGB')


            for i, pred_dict in enumerate(predictions_dicts):
                if pred_dict['bbox'] is not None:
                    for j, bbox in enumerate(pred_dict['bbox']):
                        if pred_dict['scores'][j] > score_thresh:
                            # self.debugger.add_rect((bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), img_id='input_'+str(i))
                            # self.debugger.add_3d_detection2(pred_dict['box_corners_in_fv_image'][j], c= [255, 0, 0], img_id=str(batch_imgidx[i])+'_D')
                            # self.debugger.add_point(pred_dict['box_centers_in_image'][j], c= (255, 0, 0), img_id=str(batch_imgidx[i])+'_D')
                            self.debugger.add_3d_detection2(pred_dict['box_corners_in_fv_image'][j], c=[255, 0, 0],
                                                            img_id=str(batch_imgidx[i]) + '_D')
                            # self.debugger.add_point(pred_dict['box_centers_in_image'][j], c=(255, 0, 0),
                            #                         img_id=str(batch_imgidx[i]) + '_RGB')

            if self.save_imgs:
                self.debugger.save_all_imgs(path=str(self._save_path) + '/visualize/')
            else:
                self.debugger.show_all_imgs(pause=True, stack=True, ids=example['image_idx'])

            self.debugger.remove_all_imgs()

        return predictions_dicts

    def post_process(self, dets, meta, voxel_size, scale=1):
        dets = dets.detach().cpu().numpy()
        detections = ddd_post_process(
            dets.copy(), meta, self._output_shape[3], self._output_shape[2], voxel_size, num_classes = self._num_class)
        # self.this_calib = meta['calib']
        return detections

    @property
    def avg_forward_time(self):
        return self._total_forward_time / self._total_inference_count

    @property
    def avg_postprocess_time(self):
        return self._total_postprocess_time / self._total_inference_count

    def clear_time_metrics(self):
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0

    def metrics_to_float(self):
        self.rpn_acc.float()
        self.rpn_metrics.float()
        self.rpn_cls_loss.float()
        self.rpn_loc_loss.float()
        self.rpn_total_loss.float()

    def update_metrics(self,
                       cls_loss,
                       loc_loss,
                       cls_preds,
                       labels,
                       sampled):
        batch_size = cls_preds.shape[0]
        num_class = self._num_class
        if not self._encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.view(batch_size, -1, num_class)
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        prec = prec.numpy()
        recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "cls_loss": float(rpn_cls_loss),
            "cls_loss_rt": float(cls_loss.data.cpu().numpy()),
            'loc_loss': float(rpn_loc_loss),
            "loc_loss_rt": float(loc_loss.data.cpu().numpy()),
            "rpn_acc": float(rpn_acc),
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret[f"prec@{int(thresh*100)}"] = float(prec[i])
            ret[f"rec@{int(thresh*100)}"] = float(recall[i])
        return ret

    def clear_metrics(self):
        self.rpn_acc.clear()
        self.rpn_metrics.clear()
        self.rpn_cls_loss.clear()
        self.rpn_loc_loss.clear()
        self.rpn_total_loss.clear()

    @staticmethod
    def convert_norm_to_float(net):
        '''
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        '''
        if isinstance(net, torch.nn.modules.batchnorm._BatchNorm):
            net.float()
        for child in net.children():
            VoxelNet.convert_norm_to_float(net)
        return net


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
        boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                box_code_size=7):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses


def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type=LossNormType.NormByNumPositives,
                         dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared


def assign_weight_to_each_class(labels,
                                weight_per_class,
                                norm_by_num=True,
                                dtype=torch.float32):
    weights = torch.zeros(labels.shape, dtype=dtype, device=labels.device)
    for label, weight in weight_per_class:
        positives = (labels == label).type(dtype)
        weight_class = weight * positives
        if norm_by_num:
            normalizer = positives.sum()
            normalizer = torch.clamp(normalizer, min=1.0)
            weight_class /= normalizer
        weights += weight_class
    return weights


def get_direction_target(anchors, reg_targets, one_hot=True):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, 7)
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (rot_gt > 0).long()
    if one_hot:
        dir_cls_targets = torchplus.nn.one_hot(
            dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets
