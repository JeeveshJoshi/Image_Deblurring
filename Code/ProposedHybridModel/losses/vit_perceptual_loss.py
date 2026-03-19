import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig

class ViTFeatureExtractor(nn.Module):
    def __init__(self, vit_model, feature_layer_idx):
        super().__init__()
        self.vit_model = vit_model 
        self.feature_layer_idx = feature_layer_idx
        
        for param in self.vit_model.parameters():
            param.requires_grad = False
        self.vit_model.eval()

        self.features = None
        self.hook = self.vit_model.encoder.layer[self.feature_layer_idx].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.features = output[0] 

    def forward(self, x):
        _ = self.vit_model(pixel_values=x)
        return self.features

class ViTPerceptualLoss(nn.Module):
    def __init__(self, feature_layer_idx=5):
        super().__init__()
        self.vit_base_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.feature_extractor = ViTFeatureExtractor(self.vit_base_model, feature_layer_idx)
        
        self.preprocess = lambda x: F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, deblurred_image, sharp_image):
        if deblurred_image.min() < 0 or sharp_image.min() < 0:
            deblurred_image = (deblurred_image + 1) / 2 
            sharp_image = (sharp_image + 1) / 2

        deblurred_resized = self.preprocess(deblurred_image)
        sharp_resized = self.preprocess(sharp_image)

        deblurred_norm = (deblurred_resized - self.mean) / self.std
        sharp_norm = (sharp_resized - self.mean) / self.std

        with torch.no_grad():
            features_deblurred = self.feature_extractor(deblurred_norm)
            features_sharp = self.feature_extractor(sharp_norm)

        return F.l1_loss(features_deblurred, features_sharp)