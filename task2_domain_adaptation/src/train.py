"""
Training utilities for Task 2 methods.

Source-only: cross-entropy on 3 source domains
DAN: + lambda_MMD * MMD^2
DANN: + domain loss with GRL
CDAN: + domain loss on vec(f ⊗ p) with GRL
"""

# TODO: Implement unified training loop
# Key functions:
#   train_source_only(model, source_loaders, val_loaders, ...)
#   train_dan(model, source_loaders, target_loader, val_loaders, lambda_mmd=1.0, ...)
#   train_dann(model, discriminator, source_loaders, target_loader, val_loaders, ...)
#   train_cdan(model, discriminator, source_loaders, target_loader, val_loaders, ...)
