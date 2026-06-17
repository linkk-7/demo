""" class to process superpoint net
# may be some duplication with model_wrap.py
# PointTracker is from Daniel's repo.
"""

import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from typing import Dict, Tuple
from numpy.typing import NDArray

from trained_utils.utils import flattenDetection
from trained_utils.losses import extract_patch_from_points



class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        torch.cuda.empty_cache()
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        torch.cuda.empty_cache()
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

def extract_topk_pts_and_scores(heatmap, N, x_min, x_max, y_min, y_max, conf_thresh):
    # 1. 移除不必要的 batch 和通道维度
    heatmap = heatmap.squeeze()  # 变成 torch.Size([1440, 2560])

    # 2. Apply confidence threshold mask to filter keypoints with scores > conf_thresh
    confidence_mask = heatmap > conf_thresh  # Mask for values greater than conf_thresh


    # # 2. 展平成一维，并找到 top N 的值和索引
    # topk_values, topk_indices = torch.topk(heatmap.view(-1), N)

    # # 3. 计算对应的坐标 (height, width)
    # h, w = heatmap.shape
    # topk_y = topk_indices // w  # 计算 y 坐标
    # topk_x = topk_indices % w   # 计算 x 坐标
    # topk_coords = torch.stack((topk_x, topk_y), dim=-1)  # 形状为 [N, 2]

    # # 提取 x 和 y 坐标
    # x_coords = topk_coords[:, 0]
    # y_coords = topk_coords[:, 1]

    # 3. Get coordinates of all keypoints that satisfy the confidence threshold
    y_coords, x_coords = torch.nonzero(confidence_mask, as_tuple=True)  # Get (y, x) coordinates where mask is True
    valid_scores = heatmap[confidence_mask]  # Get confidence scores for the valid points

    # 4. Create a mask based on the x and y coordinate limits (region of interest)
    # 创建条件掩码：满足 807 <= x <= 2191 且 457 <= y <= 858
    x_min_not_none = x_min
    if x_min is None:
        x_min_not_none = 0

    x_max_not_none = x_max
    if x_max is None:
        x_max_not_none = 500 

    y_min_not_none = y_min
    if y_min is None:
        y_min_not_none = 0

    y_max_not_none = y_max
    if y_max is None:
        y_max_not_none = 500

    mask = (x_coords >= x_min_not_none) & (x_coords <= x_max_not_none) & (y_coords >= y_min_not_none) & (y_coords <= y_max_not_none)


    # # 根据掩码过滤 topk_values 和 topk_coords
    # topk_values = topk_values[mask]
    # topk_coords = topk_coords[mask]

    # 5. Filter the coordinates and scores based on the region of interest
    filtered_x_coords = x_coords[mask]
    filtered_y_coords = y_coords[mask]
    filtered_scores = valid_scores[mask]

    # 6. Sort the filtered points based on their confidence scores in descending order
    sorted_scores, sorted_indices = torch.sort(filtered_scores, descending=True)  # Sort scores in descending order
    sorted_x_coords = filtered_x_coords[sorted_indices]
    sorted_y_coords = filtered_y_coords[sorted_indices]

    # # 7. Check if the number of valid points is greater than N
    # if sorted_scores.size(0) > N:
    #     # Take only the top N points
    #     sorted_x_coords = sorted_x_coords[:N]
    #     sorted_y_coords = sorted_y_coords[:N]
    #     sorted_scores = sorted_scores[:N]
        
    # 8. Stack the sorted coordinates
    sorted_coords = torch.stack((sorted_x_coords, sorted_y_coords), dim=-1)  # Shape is [N, 2]

    # print('extract_topk_pts_and_scores topk_values', topk_values, topk_values.size())
    # print('extract_topk_pts_and_scores topk_coords', topk_coords, topk_coords.size())

    # print('Sorted Scores:', sorted_scores, sorted_scores.size())
    # print('Sorted Coordinates:', sorted_coords, sorted_coords.size())

    # # 4. 转换为所需形状
    # pts = topk_coords.unsqueeze(0)  # 形状为 [1, N, 2]
    # scores = topk_values.view(1, topk_values.size()[0], 1)  # 形状为 [1, N, 1]

    # 9. Convert to the required shape
    pts = sorted_coords.unsqueeze(0)  # Shape becomes [1, N, 2]
    scores = sorted_scores.view(1, sorted_scores.size(0), 1)  # Shape becomes [1, N, 1]    

    return pts, scores

def extract_pts_desc(dense_desc, pts):
    # 1. 移除不必要的 batch 维度，得到 dense_desc 形状为 [256, 1440, 2560]
    dense_desc = dense_desc.squeeze(0)  # torch.Size([256, 1440, 2560])

    # 2. 提取点坐标
    pts = pts.squeeze(0)  # torch.Size([N, 2])

    # 3. 获取 N 个点的坐标 (h, w)
    h_coords = pts[:, 1]  # torch.Size([N])
    w_coords = pts[:, 0]  # torch.Size([N])

    # 4. 从 dense_desc 中根据坐标提取描述符，输出形状为 [N, 256]
    pts_desc = dense_desc[:, h_coords, w_coords].permute(1, 0)  # torch.Size([N, 256])

    # 5. 添加 batch 维度，返回形状为 [1, N, 256]
    pts_desc = pts_desc.unsqueeze(0)  # torch.Size([1, N, 256])

    return pts_desc

def labels2Dto3D(cell_size, labels):
    H, W = labels.shape[0], labels.shape[1]
    Hc, Wc = H // cell_size, W // cell_size
    labels = labels[:, np.newaxis, :, np.newaxis]
    labels = labels.reshape(Hc, cell_size, Wc, cell_size)
    labels = np.transpose(labels, [1, 3, 0, 2])
    labels = labels.reshape(1, cell_size ** 2, Hc, Wc)
    labels = labels.squeeze()
    dustbin = labels.sum(axis=0)
    dustbin = 1 - dustbin
    dustbin[dustbin < 0] = 0
    labels = np.concatenate((labels, dustbin[np.newaxis, :, :]), axis=0)
    return labels

def toNumpy(tensor):
    return tensor.detach().cpu().numpy()

def norm_desc(desc):
    dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
    desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return desc



class SuperPointNet_gauss2(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, subpixel_channel=1):
        super(SuperPointNet_gauss2, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        # self.down4 = down(c4, 512)
        self.up1 = up(c4+c3, c2)
        self.up2 = up(c2+c2, c1)
        self.up3 = up(c1+c1, c1)
        self.outc = outconv(c1, subpixel_channel)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.outc = outconv(64, n_classes)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)
        self.output = None


    def forward(self, x, subpixel=False):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu\
        torch.cuda.empty_cache()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)        
        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))        
        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        output = {'semi': semi, 'desc': desc}
        self.output = output        
        if subpixel:
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
            x = self.outc(x)
            output = {'semi': semi, 'desc': desc, 'x': x}
            self.output = output
            # return semi, desc, x
            return output        
        # return semi, desc
        return output
class SuperPointFrontend_torch(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """
    '''
    * SuperPointFrontend_torch:
    ** note: the input, output is different from that of SuperPointFrontend
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    desc: [batch_size, np(256, N)]
    '''

    def __init__(self, weights_path, conf_thresh: float,  nms_dist: int, max_key_points=4096,
                 cuda=False, device='cpu'):
        self.name = 'SuperPoint'
        self.cuda = cuda
        self.nms_dist = nms_dist
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.sparsemap = None
        self.heatmap = None
        self.pts = None
        self.pts_subpixel = None
        self.patches = None
        self.max_key_points = max_key_points
        self.conf_thresh = conf_thresh
        self.device=device
        self.subpixel = False
        self.params = {}

        
        self.loadModel(weights_path)


    def loadModel(self, weights_path):
        # Load the network in inference mode.
        if weights_path[-4:] != '.tar':
            return
        self.net = SuperPointNet_gauss2(**self.params)
        checkpoint = torch.load(weights_path,
                                map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.net = self.net.to(self.device)


    def net_parallel(self):
        print("=== Let's use", torch.cuda.device_count(), "GPUs!")
        self.net = nn.DataParallel(self.net)


    def nms_fast_torch(self, pts: torch.Tensor, scores: torch.Tensor, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on torch tensors:
        pts: [1, N, 2] (coordinates of keypoints)
        scores: [1, N, 1] (confidence scores of keypoints)

        Inputs:
        pts - torch.Size([1, N, 2]) tensor with [x_i, y_i] coordinates.
        scores - torch.Size([1, N, 1]) tensor with confidence scores.
        H - Image height.
        W - Image width.
        dist_thresh - Distance to suppress, measured as an infinity norm distance.

        Returns:
        nmsed_pts - torch.Size([1, M, 2]) tensor with surviving keypoints coordinates.
        nmsed_scores - torch.Size([1, M, 1]) tensor with surviving keypoints scores.
        """
        # Remove the batch dimension
        pts = pts.squeeze(0)  # [N, 2]
        scores = scores.squeeze(0)  # [N, 1]

        # print("inner pts", pts, pts.size())
        # print("inner scores", scores, scores.size())

        # Sort by confidence (scores)
        scores, indices = torch.sort(scores.squeeze(), descending=True)  # [N]
        pts = pts[indices]  # [N, 2]

        # Round the keypoints to the nearest integer
        pts_rounded = pts.round().long()  # [N, 2]

        # Edge case: no points
        if pts_rounded.shape[0] == 0:
            return torch.empty(1, 0, 2), torch.empty(1, 0, 1)

        # Edge case: only one point
        if pts_rounded.shape[0] == 1:
            return pts_rounded.unsqueeze(0), scores.unsqueeze(0).unsqueeze(0)

        # Initialize the grid
        grid = torch.zeros((H, W), dtype=torch.int)  # [H, W]
        inds = torch.zeros((H, W), dtype=torch.long)  # [H, W]

        # Mark points on the grid and store their indices
        for i, pt in enumerate(pts_rounded):
            grid[pt[1], pt[0]] = 1
            inds[pt[1], pt[0]] = i

        # Pad the grid to handle borders
        pad = dist_thresh
        grid = torch.nn.functional.pad(grid, (pad, pad, pad, pad), mode='constant', value=0)

        # Non-Max Suppression
        kept_indices = []
        length = 0
        for i, pt in enumerate(pts_rounded):
            pt_padded = (pt[0] + pad, pt[1] + pad)
            if grid[pt_padded[1], pt_padded[0]] == 1:  # If not suppressed
                # Suppress neighbors within the distance threshold
                grid[pt_padded[1] - pad: pt_padded[1] + pad + 1, pt_padded[0] - pad: pt_padded[0] + pad + 1] = 0
                grid[pt_padded[1], pt_padded[0]] = -1  # Mark as kept
                kept_indices.append(i)
                length += 1
                if length >= self.max_key_points:
                    break

        # Get surviving points and scores
        kept_indices = torch.tensor(kept_indices, dtype=torch.long)
        nmsed_pts = pts[kept_indices].unsqueeze(0)  # [1, M, 2]
        nmsed_scores = scores[kept_indices].unsqueeze(0).unsqueeze(-1)  # [1, M, 1]
        return nmsed_pts, nmsed_scores

    def getSparsemap(self):
        return self.sparsemap

    @property
    def points(self):
        # print("get pts")
        return self.pts

    @property
    def heatmap(self):
        # print("get heatmap")
        return self._heatmap

    @heatmap.setter
    def heatmap(self, heatmap):
        # print("set heatmap")
        self._heatmap = heatmap

    # @staticmethod
    def get_image_patches(self, pts, image, patch_size=5):
        """
        input:
            image: np [H, W]
        return:
            patches: np [N, patch, patch]

        """
        
        pts = pts[0].transpose().copy()
        patches = extract_patch_from_points(image, pts, patch_size=patch_size)
        patches = np.stack(patches)
        return patches

    def sample_desc_from_points(self, coarse_desc, pts):
        # --- Process descriptor.
        H, W = coarse_desc.shape[2]*self.cell, coarse_desc.shape[3]*self.cell
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            samp_pts = samp_pts.to(self.device)
            desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts, align_corners=True)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return desc


    def subpixel_predict(self, pred_res, points, verbose=False):
        """
        input:
            labels_res: numpy [2, H, W]
            points: [3, N]
        return:
            subpixels: [3, N]
        """
        points = points.T
        D = points.shape[0]
        if points.shape[1] == 0:
            pts_subpixel = np.zeros((D, 0))
        else:
            # points_res = pred_res[:,points[1,:].astype(int), points[0,:].astype(int)]
            points_res = pred_res[:, points[1, :].long(), points[0, :].long()]
            pts_subpixel = points.clone()
            if verbose: print("before: ", pts_subpixel[:,:5])
            pts_subpixel[:2,:] += points_res
            if verbose: print("after: ", pts_subpixel[:,:5])
        return pts_subpixel

    def truncate_array(self, arr):
        # 获取输入数组的列数
        M = arr.shape[1]
        
        # 如果 M > N，则截取前N列；否则返回原数组
        if M > self.max_key_points:
            return arr[:, :self.max_key_points]
        else:
            return arr

    def run(self, inp: torch.Tensor, x_min: int, x_max: int, y_min: int, y_max: int) -> Dict:
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW tensor float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
          """
        inp = inp.to(self.device)
        batch_size, H, W = inp.shape[0], inp.shape[2], inp.shape[3]
        # print("input image: ", inp, inp.size())
        with torch.no_grad():
            outs = self.net.forward(inp, subpixel=self.subpixel)
            torch.cuda.empty_cache()
            # outs = self.net.forward(inp)
            # semi, coarse_desc = outs[0], outs[1]
            semi, coarse_desc = outs['semi'], outs['desc']

        # as tensor
        
        # flatten detection
        heatmap = flattenDetection(semi, tensor=True)
        self.heatmap = heatmap
        print("heatmap: ", heatmap, heatmap.size())

        pts, scores = extract_topk_pts_and_scores(heatmap, self.max_key_points, x_min, x_max, y_min, y_max, self.conf_thresh)
        # print("Pts thresholded with confidence scores: ", pts, pts.size())
        # print("scores thresholded with confidence scores: ", scores, scores.size())
        pts, scores = self.nms_fast_torch(pts, scores, H, W, self.nms_dist)
        # print("Pts after nms_fast_torch: ", pts, pts.size(), "self.max_key_points", self.max_key_points)

        pts = pts.squeeze(0)  # [N, 2]
        scores = scores.squeeze(0)  # [N, 1]

        # 分别获取 x 和 y 坐标
        x_coords = pts[:, 0]  # [N]
        y_coords = pts[:, 1]  # [N]

        # 判断哪些点在边界内 (逻辑操作)
        toremoveW = torch.logical_or(x_coords < self.border_remove, x_coords >= (W - self.border_remove))  # [N]
        toremoveH = torch.logical_or(y_coords < self.border_remove, y_coords >= (H - self.border_remove))  # [N]
        
        # 将需要移除的点标记为 True
        toremove = torch.logical_or(toremoveW, toremoveH)  # [N]

        # 过滤掉不需要移除的点
        pts = pts[~toremove]  # [M, 2], M <= N
        scores = scores[~toremove]  # [M, 1], M <= N

        # 重新添加 batch 维度
        pts = pts.unsqueeze(0)  # [1, M, 2]
        scores = scores.unsqueeze(0)  # [1, M, 1]

        self.pts = pts
        
        if self.subpixel:
            # labels_res = outs[2]
            labels_res = outs['x']
            self.pts_subpixel = [self.truncate_array(self.subpixel_predict(toNumpy(labels_res[i, ...]), pts[i])) for i in range(batch_size)]

        dense_desc = nn.functional.interpolate(coarse_desc, scale_factor=(self.cell, self.cell), mode='bilinear')
        # norm the descriptor

        dense_desc = norm_desc(dense_desc)

        pts_desc = extract_pts_desc(dense_desc, pts)


        return {
            'keypoints': pts,
            'keypoint_scores': scores,
            'descriptors': pts_desc,
            'image_size': torch.tensor([[W, H]], dtype=torch.float32)
        }


