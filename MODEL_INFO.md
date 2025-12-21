# MRF-DETR Model Information

## Segmentation Preview

```text
Using a different number of positional encodings than DINOv2, which means we're not loading DINOv2 backbone weights. This is not a problem if finetuning a pretrained RF-DETR model.
Using patch size 12 instead of 14, which means we're not loading DINOv2 backbone weights. This is not a problem if finetuning a pretrained RF-DETR model.

encoder='dinov2_windowed_small' out_feature_indexes=[3, 6, 9, 12] dec_layers=4 two_stage=True projector_scale=['P4'] hidden_dim=256 patch_size=12 num_windows=2 sa_nheads=8 ca_nheads=16 dec_n_points=2 bbox_reparam=True lite_refpoint_refine=True layer_norm=True amp=True num_classes=90 pretrain_weights='rf-detr-seg-preview.pt' device='cpu' resolution=432 group_detr=13 gradient_checkpointing=False positional_encoding_size=36 ia_bce_loss=True cls_loss_coef=1.0 segmentation_head=True mask_downsample_ratio=4 num_queries=200 num_select=200
```

## Base

```text
Loading pretrain weights
encoder='dinov2_windowed_small' out_feature_indexes=[2, 5, 8, 11] dec_layers=3 two_stage=True projector_scale=['P4'] hidden_dim=256 patch_size=14 num_windows=4 sa_nheads=8 ca_nheads=16 dec_n_points=2 bbox_reparam=True lite_refpoint_refine=True layer_norm=True amp=True num_classes=90 pretrain_weights='rf-detr-base.pth' device='cpu' resolution=560 group_detr=13 gradient_checkpointing=False positional_encoding_size=37 ia_bce_loss=True cls_loss_coef=1.0 segmentation_head=False mask_downsample_ratio=4 num_queries=300 num_select=300
```

## Nano

```text
Using a different number of positional encodings than DINOv2, which means we're not loading DINOv2 backbone weights. This is not a problem if finetuning a pretrained RF-DETR model.
Using patch size 16 instead of 14, which means we're not loading DINOv2 backbone weights. This is not a problem if finetuning a pretrained RF-DETR model.

encoder='dinov2_windowed_small' out_feature_indexes=[3, 6, 9, 12] dec_layers=2 two_stage=True projector_scale=['P4'] hidden_dim=256 patch_size=16 num_windows=2 sa_nheads=8 ca_nheads=16 dec_n_points=2 bbox_reparam=True lite_refpoint_refine=True layer_norm=True amp=True num_classes=90 pretrain_weights='rf-detr-nano.pth' device='cpu' resolution=384 group_detr=13 gradient_checkpointing=False positional_encoding_size=24 ia_bce_loss=True cls_loss_coef=1.0 segmentation_head=False mask_downsample_ratio=4 num_queries=300 num_select=300
```

## Small

```text
Using a different number of positional encodings than DINOv2, which means we're not loading DINOv2 backbone weights. This is not a problem if finetuning a pretrained RF-DETR model.
Using patch size 16 instead of 14, which means we're not loading DINOv2 backbone weights. This is not a problem if finetuning a pretrained RF-DETR model.

encoder='dinov2_windowed_small' out_feature_indexes=[3, 6, 9, 12] dec_layers=3 two_stage=True projector_scale=['P4'] hidden_dim=256 patch_size=16 num_windows=2 sa_nheads=8 ca_nheads=16 dec_n_points=2 bbox_reparam=True lite_refpoint_refine=True layer_norm=True amp=True num_classes=90 pretrain_weights='rf-detr-small.pth' device='cpu' resolution=512 group_detr=13 gradient_checkpointing=False positional_encoding_size=32 ia_bce_loss=True cls_loss_coef=1.0 segmentation_head=False mask_downsample_ratio=4 num_queries=300 num_select=300
```

## Medium

```text
Using a different number of positional encodings than DINOv2, which means we're not loading DINOv2 backbone weights. This is not a problem if finetuning a pretrained RF-DETR model.
Using patch size 16 instead of 14, which means we're not loading DINOv2 backbone weights. This is not a problem if finetuning a pretrained RF-DETR model.

encoder='dinov2_windowed_small' out_feature_indexes=[3, 6, 9, 12] dec_layers=4 two_stage=True projector_scale=['P4'] hidden_dim=256 patch_size=16 num_windows=2 sa_nheads=8 ca_nheads=16 dec_n_points=2 bbox_reparam=True lite_refpoint_refine=True layer_norm=True amp=True num_classes=90 pretrain_weights='rf-detr-medium.pth' device='cpu' resolution=576 group_detr=13 gradient_checkpointing=False positional_encoding_size=36 ia_bce_loss=True cls_loss_coef=1.0 segmentation_head=False mask_downsample_ratio=4 num_queries=300 num_select=300
```

## Large

```text
2025-12-20 13:54:05.093701: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-12-20 13:54:05.434189: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE4.1 SSE4.2 AVX AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

encoder='dinov2_windowed_base' out_feature_indexes=[2, 5, 8, 11] dec_layers=3 two_stage=True projector_scale=['P3', 'P5'] hidden_dim=384 patch_size=14 num_windows=4 sa_nheads=12 ca_nheads=24 dec_n_points=4 bbox_reparam=True lite_refpoint_refine=True layer_norm=True amp=True num_classes=90 pretrain_weights='rf-detr-large.pth' device='cpu' resolution=560 group_detr=13 gradient_checkpointing=False positional_encoding_size=37 ia_bce_loss=True cls_loss_coef=1.0 segmentation_head=False mask_downsample_ratio=4 num_queries=300 num_select=300
```
