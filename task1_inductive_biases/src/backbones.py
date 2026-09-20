"""
Pretrained backbone wrappers for Task 1.

All backbones are FROZEN. Only linear classifier heads are trained.
- ResNet-50 (IMAGENET1K_V2) -> global-avg-pooled feature (2048-d)
- ViT-B/16  (IMAGENET1K_V1) -> final class token (768-d)
- CLIP ViT-B-32 (openai) -> normalized image embedding (512-d)
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights

try:
    import open_clip
except ImportError:
    open_clip = None

RESNET_NORM = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
VIT_NORM    = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
CLIP_NORM   = transforms.Normalize([0.48145466,0.4578275,0.40821073],[0.26862954,0.26130258,0.27577711])


class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(m.children())[:-1])
        self.dim = 2048
        self.norm = RESNET_NORM
        for p in self.features.parameters(): p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            return self.features(x).flatten(1)


class ViTB16Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.dim = 768
        self.norm = VIT_NORM
        for p in self.model.parameters(): p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.model._process_input(x)
            B = x.shape[0]
            cls = self.model.class_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)
            x = self.model.encoder(x)
            return x[:, 0]


class CLIPBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        assert open_clip, "pip install open_clip_torch"
        self.model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        self.dim = 512
        self.norm = CLIP_NORM
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        for p in self.model.parameters(): p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            f = self.model.encode_image(x)
            return f / f.norm(dim=-1, keepdim=True)

    def encode_text(self, texts):
        tokens = self.tokenizer(texts)
        with torch.no_grad():
            f = self.model.encode_text(tokens.to(next(self.model.parameters()).device))
            return f / f.norm(dim=-1, keepdim=True)


class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        return self.fc(x)
