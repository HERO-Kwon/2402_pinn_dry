import torch
import torch.nn.functional as F
from torch import nn

from layout_data.utils.initialize import initialize_weights


class _EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, inlet=False):
        super(_EncoderBlock, self).__init__()
        self.inlet = inlet
        if inlet:
            first_conv_channels = in_channels
        else:
            first_conv_channels = out_channels
        layers = [
            nn.Conv2d(first_conv_channels, out_channels, kernel_size=3, padding=1),#, padding_mode='reflect'),
            nn.SiLU(),
        ]
        self.encode = nn.Sequential(*layers)
        
        if inlet==False:
            self.down_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.SiLU()
            )

    def forward(self, x):
        if self.inlet==False:
            x = self.down_conv(x)
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),#, padding_mode='reflect'),
            nn.Upsample(scale_factor=(2,2), mode='nearest'),
        )
        self.decode2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),#, padding_mode='reflect'),
            nn.SiLU(),
        )

    def forward(self, x, x_enc):
        x1 = self.decode1(x)
        x2 = torch.cat([x1,x_enc], dim=1)
        return self.decode2(x2)

class SeparateDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, factors=2, num_classes=1):
        super(SeparateDecoder, self).__init__()
        self.dec4 = _DecoderBlock(512 * factors, 256 * factors)
        self.dec3 = _DecoderBlock(256 * factors, 128 * factors)
        self.dec2 = _DecoderBlock(128 * factors, 64 * factors)
        self.dec1 = _DecoderBlock(64 * factors, 32 * factors)
        self.final = nn.Conv2d(32 * factors, num_classes, kernel_size=1)

    def forward(self, center, x4, x3, x2, x1):
        x = self.dec4(center, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        return self.final(x)

class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=1, factors=2):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(in_channels, 32 * factors, inlet=True)
        self.enc2 = _EncoderBlock(32 * factors, 64 * factors)
        self.enc3 = _EncoderBlock(64 * factors, 128 * factors)
        self.enc4 = _EncoderBlock(128 * factors, 256 * factors)
        self.center = _EncoderBlock(256 * factors, 512 * factors)
        if num_classes==3:
            # Separate decoders for U, V, and P
            self.decoder_U = SeparateDecoder(512 * factors, 1)
            self.decoder_V = SeparateDecoder(512 * factors, 1)
            self.decoder_P = SeparateDecoder(512 * factors, 1)
        elif num_classes==2:
            # Separate decoders for T
            self.decoder_tmp = SeparateDecoder(512 * factors, 1)
            self.decoder_wc = SeparateDecoder(512 * factors, 1)
        initialize_weights(self)

    def forward(self, x, num_classes):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        if num_classes==3:
            # Decoding paths for U, V, and P
            out_U = self.decoder_U(center, enc4, enc3, enc2, enc1)
            out_V = self.decoder_V(center, enc4, enc3, enc2, enc1)
            out_P = self.decoder_P(center, enc4, enc3, enc2, enc1)
            out_Unet = torch.cat([out_U, out_V, out_P], dim=1)
        elif num_classes==2:
            out_tmp = self.decoder_tmp(center, enc4, enc3, enc2, enc1)
            out_wc = self.decoder_wc(center, enc4, enc3, enc2, enc1)
            out_Unet = torch.cat([out_tmp, out_wc], dim=1)
        return out_Unet

if __name__ == '__main__':
    model = UNet(in_channels=1, num_classes=1)
    print(model)
    x = torch.randn(1, 1, 200, 200)
    with torch.no_grad():
        final = model(x)
        print(final.shape)
