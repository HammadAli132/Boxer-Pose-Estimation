import torch.nn as nn
import torch

class DeconvolutionalHead(nn.Module):
    """
    A lightweight decoder head for Vision Transformer backbones (like DinoV2).
    It uses Deconvolution layers to upsample the coarse features into high-res heatmaps.
    """
    def __init__(self, in_channels: int, num_keypoints: int = 14):
        super().__init__()
        
        # We assume DinoV2-ViTS14 (in_channels=384) 
        # The head must perform spatial upsampling and channel regression.
        
        # Deconv Block 1 (e.g., 4x upsampling for ViT-S/14 patch size)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Deconv Block 2 (e.g., 2x upsampling)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Final prediction layer: reduces channels to N_Keypoints
        self.final_layer = nn.Conv2d(256, num_keypoints, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # Input 'x' is typically (B, Sequence_Length, C) from the ViT.
        # It must be reshaped back into a spatial grid (B, C, H, W) before deconvolution.
        
        # DinoV2 output needs reshaping: (B, L, C) -> (B, C, H, W)
        B, L, C = x.shape
        # Assuming H=W (e.g., 22x22 for ViTS14 at 308x308)
        H_feat = W_feat = int(L**0.5)
        
        # Ignore the CLS token if present (DinoV2 generally outputs all patch tokens)
        if L == (H_feat * W_feat) + 1: # If CLS token is present
             x = x[:, 1:] 
        
        x = x.transpose(1, 2).reshape(B, C, H_feat, W_feat) # (B, C, H_feat, W_feat)

        # Upsample
        x = self.deconv1(x)
        x = self.deconv2(x)
        
        # Predict heatmaps
        heatmaps = self.final_layer(x) # Output shape: (B, 14, H_out, W_out)
        return heatmaps