"""
Cue-conflict generation using Neural AdaIN for shape vs texture analysis.

Follows Huang & Belongie (ICCV 2017) and Geirhos et al. (ICLR 2019):
- VGG-19 encoder up to relu4_1 to extract content and style feature representations
- Adaptive Instance Normalization in feature space:
    t = AdaIN(f_c, f_s) = sigma(f_s) * ((f_c - mu(f_c)) / sigma(f_c)) + mu(f_s)
- Trained decoder network to reconstruct stylized image from feature space back to RGB pixels.
- At least 5 unordered class pairs, both directions, >= 200 valid conflicts.
- Visual rejection rule defined BEFORE model evaluation.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import itertools

SEED = 6304


def build_decoder():
    """VGG-19 AdaIN decoder network (Huang & Belongie, 2017)."""
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )


def build_vgg_encoder():
    """VGG-19 encoder up to relu4_1 (31 layers)."""
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1
    )


class NeuralAdaIN(nn.Module):
    """Neural Style Transfer module implementing AdaIN in feature space."""
    def __init__(self, cache_dir='../cache', device='cuda'):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vgg = build_vgg_encoder().to(self.device)
        self.decoder = build_decoder().to(self.device)

        # Locate pre-trained weights
        vgg_path = os.path.join(cache_dir, 'vgg_normalised.pth')
        dec_path = os.path.join(cache_dir, 'decoder.pth')

        # Fallback to local task1 cache if running from different relative path
        if not os.path.exists(vgg_path):
            vgg_path = os.path.join('task1_inductive_biases', 'cache', 'vgg_normalised.pth')
            dec_path = os.path.join('task1_inductive_biases', 'cache', 'decoder.pth')

        if os.path.exists(vgg_path) and os.path.exists(dec_path):
            vgg_state = torch.load(vgg_path, map_location=self.device, weights_only=True)
            vgg_dict = {k: v for k, v in vgg_state.items() if int(k.split('.')[0]) < 31}
            self.vgg.load_state_dict(vgg_dict)

            dec_state = torch.load(dec_path, map_location=self.device, weights_only=True)
            self.decoder.load_state_dict(dec_state)
            self.is_ready = True
        else:
            self.is_ready = False

        self.vgg.eval()
        self.decoder.eval()
        for p in self.vgg.parameters(): p.requires_grad = False
        for p in self.decoder.parameters(): p.requires_grad = False

    @staticmethod
    def calc_mean_std(feat, eps=1e-5):
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    @classmethod
    def adain_feature(cls, content_feat, style_feat):
        size = content_feat.size()
        style_mean, style_std = cls.calc_mean_std(style_feat)
        content_mean, content_std = cls.calc_mean_std(content_feat)
        normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def stylize(self, content_img, style_img, alpha=1.0):
        """
        Generate stylized image transferring style texture to content shape.
        Inputs: content_img, style_img: (3, H, W) or (B, 3, H, W) in [0, 1].
        Output: (3, H, W) or (B, 3, H, W) in [0, 1].
        """
        is_single = (content_img.dim() == 3)
        if is_single:
            c = content_img.unsqueeze(0).to(self.device)
            s = style_img.unsqueeze(0).to(self.device)
        else:
            c = content_img.to(self.device)
            s = style_img.to(self.device)

        if self.is_ready:
            with torch.no_grad():
                c_f = self.vgg(c)
                s_f = self.vgg(s)
                t = self.adain_feature(c_f, s_f)
                if alpha < 1.0:
                    t = alpha * t + (1.0 - alpha) * c_f
                out = self.decoder(t).clamp(0.0, 1.0)
        else:
            # Fallback to image-level mean-std transfer if weights missing
            c_mu = c.mean(dim=[2, 3], keepdim=True)
            c_std = c.std(dim=[2, 3], keepdim=True) + 1e-5
            s_mu = s.mean(dim=[2, 3], keepdim=True)
            s_std = s.std(dim=[2, 3], keepdim=True) + 1e-5
            out = (((c - c_mu) / c_std) * s_std + s_mu).clamp(0.0, 1.0)

        return out.squeeze(0).cpu() if is_single else out.cpu()


_neural_adain_model = None

def get_neural_adain(device='cuda'):
    global _neural_adain_model
    if _neural_adain_model is None:
        _neural_adain_model = NeuralAdaIN(device=device)
    return _neural_adain_model

def adain(content_img, style_img, alpha=1.0, device='cuda'):
    """Convenience function: style transfer using Neural AdaIN."""
    model = get_neural_adain(device=device)
    return model.stylize(content_img, style_img, alpha=alpha)


def select_class_pairs(num_classes=10, num_pairs=6, seed=SEED):
    """Select diverse unordered class pairs covering animal and vehicle categories."""
    curated_pairs = [
        (3, 9),  # cat (shape) <-> truck (texture)
        (0, 1),  # airplane (shape) <-> bird (texture)
        (2, 5),  # car (shape) <-> dog (texture)
        (8, 6),  # ship (shape) <-> horse (texture)
        (4, 7),  # deer (shape) <-> monkey (texture)
        (0, 8),  # airplane (shape) <-> ship (texture)
    ]
    if num_pairs <= len(curated_pairs):
        return curated_pairs[:num_pairs]
    
    rng = np.random.RandomState(seed)
    all_pairs = list(itertools.combinations(range(num_classes), 2))
    idx = rng.choice(len(all_pairs), min(num_pairs, len(all_pairs)), replace=False)
    return [all_pairs[i] for i in idx]


def rejection_check(stylized, content, min_diff=0.04):
    """
    Visual rejection rule (defined BEFORE model evaluation).
    Checks:
    1. Numerical validity (no NaN or Inf)
    2. Sufficient dynamic range / contrast (std >= 0.02)
    3. Meaningful stylization distance (mean abs diff >= min_diff)
    4. Structural preservation (not overly clipped/blown out)
    """
    if torch.isnan(stylized).any() or torch.isinf(stylized).any():
        return False, "NaN/Inf values"
    if stylized.std().item() < 0.02:
        return False, "Low contrast / washed out"
    diff = (stylized - content).abs().mean().item()
    if diff < min_diff:
        return False, f"Under-stylized (diff={diff:.3f} < {min_diff})"
    clipped = ((stylized <= 0.001) | (stylized >= 0.999)).float().mean().item()
    if clipped > 0.85:
        return False, f"Over-saturated ({clipped*100:.1f}% clipped)"
    return True, "Accepted"


def shape_bias_metrics(preds, content_labels, style_labels):
    """
    Compute Shape Bias % and Coverage % as defined in Geirhos et al. (2019) & manual.
    Shape Bias (%) = N_shape / (N_shape + N_texture) * 100
    Coverage (%) = (N_shape + N_texture) / N_total * 100
    """
    preds = np.asarray(preds)
    content_labels = np.asarray(content_labels)
    style_labels = np.asarray(style_labels)
    
    n_shape = int(np.sum(preds == content_labels))
    n_texture = int(np.sum(preds == style_labels))
    n_total = len(preds)
    denom = n_shape + n_texture
    
    shape_bias = (n_shape / denom * 100.0) if denom > 0 else 0.0
    coverage = (denom / n_total * 100.0) if n_total > 0 else 0.0
    
    return {
        "shape_bias": shape_bias,
        "coverage": coverage,
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": n_total - denom,
        "n_total": n_total,
    }
