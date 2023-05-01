import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

class REBNCONV(nn.Module): 
    def __init__(self, in_ch=3, out_ch=3, dilate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dilate, dilation=1 * dilate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout
    
def save_attention_maps(psi_values, output_dir='attention_maps_rgb'):
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Count the number of existing files in the output directory
    existing_files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    start_index = len(existing_files) + 1

    for psi in psi_values:
        # Normalize psi values to the range [0, 1]
        psi_normalized = (psi - psi.min()) / (psi.max() - psi.min())

        # Convert the tensor to a NumPy array
        psi_np = psi_normalized.squeeze().detach().cpu().numpy()

        # Convert grayscale to RGB
        psi_rgb = np.stack((psi_np,)*3, axis=-1)

        # Save the attention map as an image
        plt.imsave(os.path.join(output_dir, f'attention_map_{start_index + 1}.png'), psi_rgb)

class Attention_modified(nn.Module):
    def __init__(self, chan_x, chan_g):
        super(Attention_modified, self).__init__()
        self.W_x = nn.Conv2d(chan_x, chan_x, kernel_size=2, stride=2, padding=0, bias=True)

        self.W_g = nn.Conv2d(chan_g, chan_g, kernel_size=1, stride=1, padding=0, bias=True)

        self.psi = nn.Sequential(
            nn.Conv2d(chan_x, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.i=0

    def forward(self, x, g):
        input_size = x.size() 
        batch_size = input_size[0]
        assert batch_size == g.size(0) 

        g1 = self.W_g(g)
        x1 = self.W_x(x)  # Downscaling x width and length by 2
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        psi_up = F.interpolate(psi, size=input_size[2:], mode='bilinear', align_corners=True)  # upsample psi to make it smae length and width as x
        if self.i % 1000 == 0:
            save_attention_maps(psi)
        self.i= self.i+1
        return psi_up * x

# upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(input=src, size=tar.shape[2:], mode='bilinear', align_corners=True)
    return src
# RSU-7
class RSU7(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dilate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dilate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dilate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-6
class RSU6(nn.Module):  # UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dilate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dilate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dilate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-5
class RSU5(nn.Module):  # UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dilate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dilate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


#  RSU-4
class RSU4(nn.Module):  # UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dilate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


# RSU-4F
class RSU4F(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dilate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dilate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dilate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dilate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dilate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dilate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dilate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dilate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class AttU2NetOutside(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(AttU2NetOutside, self).__init__()

        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.att5 = Attention_modified(512, 512)
        self.stage5d = RSU4F(1024, 256, 512)
        self.att4 = Attention_modified(512, 512)
        self.stage4d = RSU4(1024, 128, 256) 
        self.att3 = Attention_modified(256, 256)
        self.stage3d = RSU5(512, 64, 128)  
        self.att2 = Attention_modified(128, 128)
        self.stage2d = RSU6(256, 32, 64) 
        self.att1 = Attention_modified(64, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx) 
        hx = self.pool12(hx1) 

        # stage 2
        hx2 = self.stage2(hx) 
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx) 
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx) 
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx) 
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx) 
        
        # -------------------- decoder --------------------
        a5 = self.att5(g=hx6, x=hx5)
        hx6up = _upsample_like(hx6, hx5) 
        hx5d = self.stage5d(torch.cat((hx6up, a5), 1)) 

        a4 = self.att4(g=hx5d, x=hx4) 
        hx5dup = _upsample_like(hx5d, hx4) 
        hx4d = self.stage4d(torch.cat((a4, hx5dup), 1)) 

        a3 = self.att3(g=hx4d, x=hx3)
        hx4dup = _upsample_like(hx4d, hx3)
        hx3d = self.stage3d(torch.cat((a3, hx4dup), 1))

        a2 = self.att2(g=hx3d, x=hx2)
        hx3dup = _upsample_like(hx3d, hx2)
        hx2d = self.stage2d(torch.cat((a2, hx3dup), 1))

        a1 = self.att1(g=hx2d, x=hx1)
        hx2dup = _upsample_like(hx2d, hx1) 
        hx1d = self.stage1d(torch.cat((a1, hx2dup), 1)) 

        # side output 
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1)) 

        return torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2), torch.sigmoid(d3), torch.sigmoid(
            d4), torch.sigmoid(d5), torch.sigmoid(d6)
