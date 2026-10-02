import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)  # (1,1,3,3)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):

        B, C, H, W = x.shape
        x_reshape = x.reshape(B * C, 1, H, W)
        sobelx = F.conv2d(x_reshape, self.weightx, padding=1)
        sobely = F.conv2d(x_reshape, self.weighty, padding=1)
        grad = torch.abs(sobelx) + torch.abs(sobely)
        grad = grad.reshape(B, C, H, W)
        return grad


def rgb_to_ycbcr(image):

    r = image[:, 0:1, :, :]
    g = image[:, 1:2, :, :]
    b = image[:, 2:3, :, :]

    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 0.5
    cr =  0.500 * r - 0.419 * g - 0.081 * b + 0.5

    return y, cb, cr


class FusionLoss(nn.Module):


    def __init__(self, alpha_intensity=1.0, alpha_gradient=0.1, alpha_color=0.5):
        super(FusionLoss, self).__init__()
        self.sobelconv = Sobelxy()
        self.alpha_intensity = alpha_intensity
        self.alpha_gradient = alpha_gradient
        self.alpha_color = alpha_color

    def forward(self, fused, ir, vis_gt):
 

        fused_y, fused_cb, fused_cr = rgb_to_ycbcr(fused)      # 各 (B, 1, H, W)
        vis_gt_y, vis_gt_cb, vis_gt_cr = rgb_to_ycbcr(vis_gt)  # 各 (B, 1, H, W)
        ir_y = ir  # (B, 1, H, W)

        intensity_target = torch.max(vis_gt_y, ir_y)  # (B, 1, H, W)
        loss_intensity = F.l1_loss(fused_y, intensity_target)

        grad_fused_y = self.sobelconv(fused_y)    # (B, 1, H, W)
        grad_vis_gt_y = self.sobelconv(vis_gt_y)  # (B, 1, H, W)
        grad_ir_y = self.sobelconv(ir_y)          # (B, 1, H, W)

        grad_target = torch.max(grad_vis_gt_y, grad_ir_y)  # (B, 1, H, W)
        loss_gradient = F.l1_loss(grad_fused_y, grad_target)

        loss_color = F.l1_loss(fused_cb, vis_gt_cb) + F.l1_loss(fused_cr, vis_gt_cr)

        loss_total = (self.alpha_intensity * loss_intensity +
                      self.alpha_gradient * loss_gradient +
                      self.alpha_color * loss_color)
        return loss_total

