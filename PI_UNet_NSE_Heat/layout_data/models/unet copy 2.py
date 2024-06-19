import torch
import torch.nn.functional as F
from torch import nn

from layout_data.utils.initialize import initialize_weights


class _EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2, padding=1, padding_mode='reflect'),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x1 = self.decode(x1)
        x = torch.cat([x2,x1],dim=1)
        return x

class _SeparateDecoder(nn.Module):
    def __init__(self, in_channels, out_channels,factors=2, num_classes=3):
        super(_SeparateDecoder, self).__init__()
        self.dec4 = _DecoderBlock(512 * factors, 128 * factors)
        self.dec3 = _DecoderBlock(256 * factors, 64 * factors)
        self.dec2 = _DecoderBlock(128 * factors, 32 * factors)
        self.dec1 = _DecoderBlock(64 * factors, 32 * factors)
        self.final = nn.Conv2d(64 * factors, num_classes, kernel_size=1)

    def forward(self, center, enc4, enc3, enc2, enc1):
        dec4 = self.dec4(center,enc4)
        dec3 = self.dec3(dec4,enc3)
        dec2 = self.dec2(dec3,enc2)
        dec1 = self.dec1(dec2,enc1)
        final = self.final(dec1)

class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=1, factors=2):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(in_channels, 32 * factors)
        self.enc2 = _EncoderBlock(32 * factors, 64 * factors)
        self.enc3 = _EncoderBlock(64 * factors, 128 * factors)
        self.enc4 = _EncoderBlock(128 * factors, 256 * factors)
        self.enc5 = _EncoderBlock(256 * factors, 256 * factors)
        #self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        #self.center = _DecoderBlock(256 * factors, 512 * factors, 256 * factors, bn=bn)
        self.decoder_U = _SeparateDecoder(128 * factors, 32 * factors)
        self.decoder_V = _SeparateDecoder(128 * factors, 32 * factors)
        self.decoder_P = _SeparateDecoder(128 * factors, 32 * factors)

        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.enc5(enc4)
        #center = self.center(self.polling(enc4))
        final_U = self.decoder_U(center, enc4, enc3, enc2, enc1)
        final_V = self.decoder_V(center, enc4, enc3, enc2, enc1)
        final_P = self.decoder_P(center, enc4, enc3, enc2, enc1)
        return torch.cat([final_U, final_V, final_P],dim=1)

if __name__ == '__main__':
    model = UNet(in_channels=1, num_classes=1)
    print(model)
    x = torch.randn(1, 1, 200, 200)
    with torch.no_grad():
        final = model(x)
        print(final.shape)
