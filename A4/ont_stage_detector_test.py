import os
import time

import matplotlib.pyplot as plt
import torch
import torchvision

from a4_helper import *
from eecs598 import reset_seed
from eecs598.grad import rel_error

from common import DetectorBackboneWithFPN
from one_stage_detector import FCOSPredictionNetwork
from common import get_fpn_location_coords
from one_stage_detector import fcos_get_deltas_from_locations, fcos_apply_deltas_to_locations
from one_stage_detector import fcos_make_centerness_targets

import random
from one_stage_detector import fcos_match_locations_to_gt

from torch.nn import functional as F

# for plotting
plt.rcParams["figure.figsize"] = (10.0, 8.0)  # set default size of plots
plt.rcParams["font.size"] = 16
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

# # To download the dataset
# !pip install wget

# # for mAP evaluation
# !rm -rf mAP
# !git clone https://github.com/Cartucho/mAP.git
# !rm -rf mAP/input/*

import multiprocessing

# Set a few constants related to data loading.
NUM_CLASSES = 20
BATCH_SIZE = 16
IMAGE_SHAPE = (224, 224)
INPUT_PATH = "/Users/fushimi/code/EECS498/A4"
DEVICE = "mps"

if __name__ == '__main__':
    NUM_WORKERS = multiprocessing.cpu_count()

    from a4_helper import VOC2007DetectionTiny

    # NOTE: Set `download=True` for the first time when you set up Google Drive folder.
    # Turn it back to `False` later for faster execution in the future.
    # If this hangs, download and place data in your drive manually as shown above.
    DATA_PATH = "/Users/fushimi/code/EECS498"
    train_dataset = VOC2007DetectionTiny(
        DATA_PATH, "train", image_size=IMAGE_SHAPE[0],
        download=False
    )
    val_dataset = VOC2007DetectionTiny(DATA_PATH, "val", image_size=IMAGE_SHAPE[0])

    print(f"Dataset sizes: train ({len(train_dataset)}), val ({len(val_dataset)})")

    # Convert individual images (JPEG) and annotations (XML files) into batches of tensors
    # `pin_memory` speeds up CPU-GPU batch transfer, `num_workers=NUM_WORKERS` loads data
    # on the main CPU process, suitable for Colab.
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS
    )

    # Use batch_size = 1 during inference - during inference we do not center crop
    # the image to detect all objects, hence they may be of different size. It is
    # easier and less redundant to use batch_size=1 rather than zero-padding images.
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, pin_memory=True, num_workers=NUM_WORKERS
    )

    train_loader_iter = iter(train_loader)
    image_paths, images, gt_boxes = next(train_loader_iter)

    print(f"image paths           : {image_paths}")
    print(f"image batch has shape : {images.shape}")
    print(f"gt_boxes has shape    : {gt_boxes.shape}")

    print(f"Five boxes per image  :")
    print(gt_boxes[:, :5, :])

    # Visualize dataset
    from torchvision import transforms
    from eecs598.utils import detection_visualizer

    # Define an "inverse" transform for the image that un-normalizes by ImageNet color
    # and mean. Without this, the images will NOT be visually understandable.
    inverse_norm = transforms.Compose(
        [
            transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ]
    )

    for idx, (_, image, gt_boxes) in enumerate(train_dataset):
        # Stop after visualizing three images.
        if idx > 2:
            break

        # Un-normalize image to bring in [0, 1] RGB range.
        image = inverse_norm(image)

        # Remove padded boxes from visualization.
        is_valid = gt_boxes[:, 4] >= 0
        detection_visualizer(image, val_dataset.idx_to_class, gt_boxes[is_valid])

    # --- Use a small RegNetX-400MF as the backbone ---
    backbone = DetectorBackboneWithFPN(out_channels=64)

    # --- Add FPN modules ---
    print("Extra FPN modules added:")
    print(backbone.fpn_params)

    # Pass a batch of dummy images (random tensors) in NCHW format and observe the output.
    dummy_images = torch.randn(2, 3, 224, 224)

    # Collect dummy output.
    dummy_fpn_feats = backbone(dummy_images)

    print(f"For dummy input images with shape: {dummy_images.shape}")
    for level_name, feat in dummy_fpn_feats.items():
        print(f"Shape of {level_name} features: {feat.shape}")

    # --- Add Head ---
    # Tiny head with `in_channels` as FPN output channels in prior cell,
    # and two conv layers in stem.
    pred_net = FCOSPredictionNetwork(
        num_classes=NUM_CLASSES, in_channels=64, stem_channels=[64, 64]
    )

    print("FCOS prediction network parameters:")
    print(pred_net)

    # Pass the dummy output from FPN (obtained in previous cell) to the head.
    dummy_preds = pred_net(dummy_fpn_feats)

    pred_cls_logits, pred_boxreg_deltas, pred_ctr_logits = dummy_preds

    print("Classification logits:")
    for level_name, feat in pred_cls_logits.items():
        print(f"Shape of {level_name} predictions: {feat.shape}")

    print("Box regression deltas:")
    for level_name, feat in pred_boxreg_deltas.items():
        print(f"Shape of {level_name} predictions: {feat.shape}")

    print("Centerness logits:")
    for level_name, feat in pred_ctr_logits.items():
        print(f"Shape of {level_name} predictions: {feat.shape}")

    # --- get_fpn_location_coords ---
    # Get shapes of each FPN level feature map. We don't call these "dummy" because
    # they don't depend on the _values_ of features, but rather only shapes.
    fpn_feats_shapes = {
        level_name: feat.shape for level_name, feat in dummy_fpn_feats.items()
    }

    # Get CPU tensors for this sanity check: (you can pass `device=` argument.
    locations_per_fpn_level = get_fpn_location_coords(fpn_feats_shapes, backbone.fpn_strides)

    # First five location co-ordinates for each feature maps.
    expected_locations = {
        "p3": torch.tensor([[4.0, 4.0], [4.0, 12.0], [4.0, 20.0], [4.0, 28.0], [4.0, 36.0]]),
        "p4": torch.tensor([[8.0, 8.0], [8.0, 24.0], [8.0, 40.0], [8.0, 56.0], [8.0, 72.0]]),
        "p5": torch.tensor([[16.0, 16.0], [16.0, 48.0], [16.0, 80.0], [16.0, 112.0], [16.0, 144.0]]),
    }

    print("First five locations per FPN level (absolute image co-ordinates):")
    for level_name, locations in locations_per_fpn_level.items():
        print(f"{level_name}: {locations[:5, :].tolist()}")
        print("rel error: ", rel_error(expected_locations[level_name], locations[:5, :]))
    
    # Visualize all the locations on first image from training data.
    for level_name, locations in locations_per_fpn_level.items():
        # Un-normalize image to bring in [0, 1] RGB range.
        image = inverse_norm(val_dataset[0][1])

        print("*" * 80)
        print(f"All locations of the image FPN level = {level_name}")
        print(f"stride = {backbone.fpn_strides[level_name]}")
        eecs598.utils.detection_visualizer(image, val_dataset.idx_to_class, points=locations.tolist())

    # --- Matching feature map locations with GT boxes ---
    # Get an image and its GT boxes from train dataset.
    _, image, gt_boxes = train_dataset[0]

    # Dictionary with keys {"p3", "p4", "p5"} and values as `(N, 5)` tensors
    # giving matched GT boxes.
    matched_boxes_per_fpn_level = fcos_match_locations_to_gt(
        locations_per_fpn_level, backbone.fpn_strides, gt_boxes
    )

    # Visualize one selected location (yellow point) and its matched GT box (red).
    # Get indices of matched locations (whose class ID is not -1) from P3 level.
    FPN_LEVEL = "p4"
    fg_idxs_p3 = (matched_boxes_per_fpn_level[FPN_LEVEL][:, 4] != -1).nonzero()

    # NOTE: Run this cell multiple times to see different matched points. For car
    # image, p3/5 will not work because the one and only box was already assigned
    # to p4 due to its compatible size to p4 stride.
    _idx = random.choice(fg_idxs_p3)

    eecs598.utils.detection_visualizer(
        inverse_norm(image),
        val_dataset.idx_to_class,
        bbox=matched_boxes_per_fpn_level[FPN_LEVEL][_idx],
        points=locations_per_fpn_level[FPN_LEVEL][_idx]
    )


    # --- GT Targets for box regression ---
    # Three hard-coded input boxes and three points lying inside them.
    # Add a dummy class ID = 1 indicating foreground
    input_boxes = torch.Tensor(
        [[10, 15, 100, 115, 1], [30, 20, 40, 30, 1], [120, 100, 200, 200, 1]]
    )
    input_locations = torch.Tensor([[30, 40], [32, 29], [125, 150]])

    # Here we do a simple sanity check - getting deltas for a particular set of boxes
    # and applying them back to centers should give us the same boxes. Setting a random
    # stride = 8, it should not affect reconstruction if it is same on both sides.
    _deltas = fcos_get_deltas_from_locations(input_locations, input_boxes, stride=8)
    output_boxes = fcos_apply_deltas_to_locations(_deltas, input_locations, stride=8)

    print(f"Output_boxes is:{output_boxes}")
    print(f"Deltas is {_deltas}")
    print("Rel error in reconstructed boxes:", rel_error(input_boxes[:, :4], output_boxes))


    # Another check: deltas for GT class label = -1 should be -1.
    background_box = torch.Tensor([[-1, -1, -1, -1, -1]])
    input_location = torch.Tensor([[100, 200]])

    _deltas = fcos_get_deltas_from_locations(input_location, background_box, stride=8)
    output_box = fcos_apply_deltas_to_locations(_deltas, input_location, stride=8)

    print("Background deltas should be all -1    :", _deltas)

    # Output box should be the location itself ([100, 200, 100, 200])
    print("Output box with background deltas     :", output_box)


    # --- GT targets for centerness regression ---
    # Three hard-coded input boxes and three points lying inside them.
    # Add a dummy class ID = 1 indicating foreground
    input_boxes = torch.Tensor(
        [
            [10, 15, 100, 115, 1],
            [30, 20, 40, 30, 1],
            [-1, -1, -1, -1, -1]  # background
        ]
    )
    input_locations = torch.Tensor([[30, 40], [32, 29], [125, 150]])

    expected_centerness = torch.Tensor([0.30860671401, 0.1666666716, -1.0])

    _deltas = fcos_get_deltas_from_locations(input_locations, input_boxes, stride=8)
    centerness = fcos_make_centerness_targets(_deltas)
    print(f"Centerness is {centerness}")
    print("Rel error in centerness:", rel_error(centerness, expected_centerness))


    # --- Loss ---
    from torchvision.ops import sigmoid_focal_loss


    # Sanity check: dummy model predictions for TWO locations, and
    # NUM_CLASSES = 5 (typically there are thousands of locations
    # across all FPN levels).
    # shape: (batch_size, num_locations, num_classes)
    dummy_pred_cls_logits = torch.randn(1, 2, 5)

    # Corresponding one-hot vectors of GT class labels (2, -1), one
    # foreground and one background.
    # shape: (batch_size, num_locations, num_classes)
    dummy_gt_classes = torch.Tensor([[[0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]])

    # This loss expects logits, not probabilities (DO NOT apply sigmoid!)
    cls_loss = sigmoid_focal_loss(
        inputs=dummy_pred_cls_logits, targets=dummy_gt_classes
    )
    print("Classification loss (dummy inputs/targets):")
    print(cls_loss)

    print(f"Total classification loss (un-normalized): {cls_loss.sum()}")


    from one_stage_detector import fcos_get_deltas_from_locations

    # Sanity check: dummy model predictions for TWO locations, and
    # NUM_CLASSES = 2 (typically there are thousands of locations
    # across all FPN levels).
    # Think of these as first two locations locations of "p5" level.
    dummy_locations = torch.Tensor([[32, 32], [64, 32]])
    dummy_gt_boxes = torch.Tensor(
        [
            [1, 2, 40, 50, 2],
            [-1, -1, -1, -1, -1]  # Same GT classes as above cell.
        ]
    )
    # Centerness is just a dummy value:
    dummy_gt_centerness = torch.Tensor([0.6, -1])

    # shape: (batch_size, num_locations, 4 or 1)
    dummy_pred_boxreg_deltas = torch.randn(1, 2, 4)
    dummy_pred_ctr_logits = torch.randn(1, 2, 1)

    # Collapse batch dimension.
    dummy_pred_boxreg_deltas = dummy_pred_boxreg_deltas.view(-1, 4)
    dummy_pred_ctr_logits = dummy_pred_ctr_logits.view(-1)

    # First calculate box reg loss, comparing predicted boxes and GT boxes.
    dummy_gt_deltas = fcos_get_deltas_from_locations(
        dummy_locations, dummy_gt_boxes, stride=32
    )
    # Multiply with 0.25 to average across four LTRB components.
    loss_box = 0.25 * F.l1_loss(
        dummy_pred_boxreg_deltas, dummy_gt_deltas, reduction="none"
    )

    # No loss for background:
    loss_box[dummy_gt_deltas < 0] *= 0.0
    print("Box regression loss (L1):", loss_box)


    # Now calculate centerness loss.
    centerness_loss = F.binary_cross_entropy_with_logits(
        dummy_pred_ctr_logits, dummy_gt_centerness, reduction="none"
    )
    # No loss for background:
    centerness_loss[dummy_gt_centerness < 0] *= 0.0
    print("Centerness loss (BCE):", centerness_loss)

    # In the expected loss, the first value will be different everytime due to random dummy
    # predictions. But the second value should always be zero - corresponding to background


    # --- Overfit small data ---
    from a4_helper import train_detector
    from one_stage_detector import FCOS

    reset_seed(0)

    # Take equally spaced examples from training dataset to make a subset.
    small_dataset = torch.utils.data.Subset(
        train_dataset,
        torch.linspace(0, len(train_dataset) - 1, steps=BATCH_SIZE * 10).long()
    )
    
    small_train_loader = torch.utils.data.DataLoader(
        small_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=NUM_WORKERS
    )

    detector = FCOS(
        num_classes=NUM_CLASSES,
        fpn_channels=64,
        stem_channels=[64, 64],
    )
    detector = detector.to(DEVICE)

    train_detector(
        detector,
        small_train_loader,
        learning_rate=5e-3,
        max_iters=500,
        log_period=20,
        device=DEVICE,
    )

    # After you've trained your model, save the weights for submission.
    weights_path = os.path.join(INPUT_PATH, "fcos_detector.pt")
    torch.save(detector.state_dict(), weights_path)
    # Results:
    # [Iter 0][loss: 3.262][loss_cls: 0.699][loss_box: 2.135][loss_ctr: 0.429]
    # [Iter 20][loss: 5.073][loss_cls: 1.112][loss_box: 3.281][loss_ctr: 0.680]
    # [Iter 40][loss: 3.991][loss_cls: 1.184][loss_box: 2.085][loss_ctr: 0.723]
    # [Iter 60][loss: 3.637][loss_cls: 1.053][loss_box: 1.862][loss_ctr: 0.722]
    # [Iter 80][loss: 3.102][loss_cls: 0.701][loss_box: 1.681][loss_ctr: 0.720]
    # [Iter 100][loss: 2.929][loss_cls: 0.616][loss_box: 1.592][loss_ctr: 0.721]
    # [Iter 120][loss: 2.971][loss_cls: 0.599][loss_box: 1.657][loss_ctr: 0.715]
    # [Iter 140][loss: 3.169][loss_cls: 1.002][loss_box: 1.455][loss_ctr: 0.713]
    # [Iter 160][loss: 3.358][loss_cls: 0.803][loss_box: 1.850][loss_ctr: 0.705]
    # [Iter 180][loss: 2.998][loss_cls: 0.796][loss_box: 1.505][loss_ctr: 0.696]
    # [Iter 200][loss: 3.006][loss_cls: 0.999][loss_box: 1.327][loss_ctr: 0.679]
    # [Iter 220][loss: 2.992][loss_cls: 0.739][loss_box: 1.581][loss_ctr: 0.672]
    # [Iter 240][loss: 3.424][loss_cls: 1.100][loss_box: 1.641][loss_ctr: 0.682]
    # [Iter 260][loss: 2.893][loss_cls: 0.961][loss_box: 1.263][loss_ctr: 0.669]
    # [Iter 280][loss: 2.521][loss_cls: 0.618][loss_box: 1.224][loss_ctr: 0.679]
    # [Iter 300][loss: 3.790][loss_cls: 0.975][loss_box: 2.034][loss_ctr: 0.780]
    # [Iter 320][loss: 3.398][loss_cls: 0.919][loss_box: 1.732][loss_ctr: 0.747]
    # [Iter 340][loss: 3.170][loss_cls: 0.860][loss_box: 1.599][loss_ctr: 0.710]
    # [Iter 360][loss: 2.866][loss_cls: 0.689][loss_box: 1.478][loss_ctr: 0.699]
    # [Iter 380][loss: 2.710][loss_cls: 0.643][loss_box: 1.375][loss_ctr: 0.692]
    # [Iter 400][loss: 2.612][loss_cls: 0.630][loss_box: 1.295][loss_ctr: 0.687]
    # [Iter 420][loss: 2.535][loss_cls: 0.616][loss_box: 1.238][loss_ctr: 0.681]
    # [Iter 440][loss: 2.463][loss_cls: 0.599][loss_box: 1.188][loss_ctr: 0.676]
    # [Iter 460][loss: 2.423][loss_cls: 0.588][loss_box: 1.162][loss_ctr: 0.673]
    # [Iter 480][loss: 2.412][loss_cls: 0.585][loss_box: 1.154][loss_ctr: 0.673]


    # --- Train a net ---
    from a4_helper import train_detector
    from one_stage_detector import FCOS

    reset_seed(0)

    # Slightly larger detector than in above cell.
    detector = FCOS(
        num_classes=NUM_CLASSES,
        fpn_channels=128,
        stem_channels=[128, 128],
    )
    detector = detector.to(DEVICE)

    train_detector(
        detector,
        train_loader,
        learning_rate=8e-3,
        max_iters=1000,
        log_period=100,
        device=DEVICE,
    )

    # After you've trained your model, save the weights for submission.
    weights_path = os.path.join(INPUT_PATH, "fcos_detector.pt")
    torch.save(detector.state_dict(), weights_path)


    # --- Non-Maximum Suppression (NMS) ---
    # Perform imports here to make this cell runnble independently,
    # students are likely to spend good mount of time here and it is
    # best to not require execution of prior cells.
    from common import nms

    reset_seed(0)

    DEVICE = 'mps'
    boxes = (100.0 * torch.rand(5000, 4)).round()
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0] + 1.0
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1] + 1.0
    scores = torch.randn(5000)

    names = ["your_cpu", "torchvision_cpu", "torchvision_cuda"]
    iou_thresholds = [0.3, 0.5, 0.7]
    elapsed = dict(zip(names, [0.0] * len(names)))
    intersects = dict(zip(names[1:], [0.0] * (len(names) - 1)))

    for iou_threshold in iou_thresholds:
        tic = time.time()
        my_keep = nms(boxes, scores, iou_threshold)
        elapsed["your_cpu"] += time.time() - tic

        tic = time.time()
        tv_keep = torchvision.ops.nms(boxes, scores, iou_threshold)
        elapsed["torchvision_cpu"] += time.time() - tic
        intersect = len(set(tv_keep.tolist()).intersection(my_keep.tolist())) / len(tv_keep)
        intersects["torchvision_cpu"] += intersect

        tic = time.time()
        tv_cuda_keep = torchvision.ops.nms(boxes.to(device=DEVICE), scores.to(device=DEVICE), iou_threshold).to(
            my_keep.device
        )
        torch.mps.synchronize()
        elapsed["torchvision_cuda"] += time.time() - tic
        intersect = len(set(tv_cuda_keep.tolist()).intersection(my_keep.tolist())) / len(
            tv_cuda_keep
        )
        intersects["torchvision_cuda"] += intersect

    for key in intersects:
        intersects[key] /= len(iou_thresholds)

    # You should see < 1% difference
    print("Testing NMS:")
    print("Your        CPU  implementation: %fs" % elapsed["your_cpu"])
    print("torchvision CPU  implementation: %fs" % elapsed["torchvision_cpu"])
    print("torchvision CUDA implementation: %fs" % elapsed["torchvision_cuda"])
    print("Speedup CPU : %fx" % (elapsed["your_cpu"] / elapsed["torchvision_cpu"]))
    print("Speedup CUDA: %fx" % (elapsed["your_cpu"] / elapsed["torchvision_cuda"]))
    print(
        "Difference CPU : ", 1.0 - intersects["torchvision_cpu"]
    )  # in the order of 1e-3 or less
    print(
        "Difference CUDA: ", 1.0 - intersects["torchvision_cuda"]
    )  # in the order of 1e-3 or less


    # --- Inference ---
    from a4_helper import inference_with_detector
    from one_stage_detector import FCOS


    weights_path = os.path.join(INPUT_PATH, "fcos_detector.pt")

    # Re-initialize so this cell is independent from prior cells.
    detector = FCOS(
        num_classes=NUM_CLASSES, fpn_channels=128, stem_channels=[128, 128]
    )
    detector.to(device=DEVICE)
    detector.load_state_dict(torch.load(weights_path, map_location="cpu"))

    # Prepare a small val daataset for inference:
    small_dataset = torch.utils.data.Subset(
        val_dataset,
        torch.linspace(0, len(val_dataset) - 1, steps=10).long()
    )
    small_val_loader = torch.utils.data.DataLoader(
        small_dataset, batch_size=1, pin_memory=True, num_workers=NUM_WORKERS
    )

    inference_with_detector(
        detector,
        val_loader,
        val_dataset.idx_to_class,
        score_thresh=0.4,
        nms_thresh=0.6,
        device=DEVICE,
        dtype=torch.float32,
        output_dir="mAP/input",
    )

    # !cd mAP && python main.py

    # # This script outputs an image containing per-class AP. Display it here:
    # from IPython.display import Image
    # Image(filename="/Users/fushimi/code/EECS498/mAP/output/mAP.png")  