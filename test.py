import os
import argparse
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from kornia.metrics import psnr, ssim
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

from models.WIP import PatchedWatermarkModel


def inverse_normalize(image):
    return (image / 2 + 0.5).clamp(0, 1)


def bit_acc(message_gt, message_pred):
    orig_msg = (message_gt > 0.5).int()
    recv_msg = (message_pred > 0.5).int()
    matches = (orig_msg == recv_msg).sum().item()
    total = orig_msg.numel()
    return matches / total


def iou_score(mask_gt, mask_pred):
    true_bin = (mask_gt > 0.5)
    pred_bin = (mask_pred > 0.5)
    intersection = torch.logical_and(pred_bin, true_bin).sum().float()
    union = torch.logical_or(pred_bin, true_bin).sum().float()
    iou = (intersection / (union + 1e-8)).item()
    return iou


def process_images(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_config = OmegaConf.load(args.config)
    jnd_factor = args.jnd_factor

    # Transforms
    transform = Compose([
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    mask_transform = Compose([ToTensor()])

    # Load model
    wip = PatchedWatermarkModel(**model_config)
    wip.load_state_dict(torch.load(args.checkpoint, map_location='cpu'), strict=True)
    wip = wip.to(device)
    wip.eval()

    os.makedirs(args.wm_image_path, exist_ok=True)
    os.makedirs(args.msg_save_path, exist_ok=True)
    os.makedirs(args.mask_pred_path, exist_ok=True)

    acc_sum = iou_sum = PSNR_sum = SSIM_sum = 0
    image_files = [f for f in os.listdir(args.image_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for filename in image_files:
        base = os.path.splitext(filename)[0]
        image_path = os.path.join(args.image_dir, filename)
        mask_path = os.path.join(args.mask_dir, filename)
        if not os.path.exists(mask_path):
            mask_path = os.path.join(args.mask_dir, base + '.png')

        # Load images
        image = Image.open(image_path).convert('RGB')
        W, H = image.size
        mask = Image.open(mask_path).convert('L')

        image = transform(image).unsqueeze(0).to(device)
        mask = mask_transform(mask).unsqueeze(0).to(device)

        # Random message
        message = torch.randint(0, 2, (1, model_config["message_length"]),
                                dtype=torch.float).to(device)
        np.save(os.path.join(args.msg_save_path, f"{base}.npy"), message.cpu().numpy())

        # Encode watermark
        image_256 = F.interpolate(image, size=(256, 256), mode="bilinear")
        mask_256 = F.interpolate(mask, size=(256, 256), mode="bilinear")
        mask_gt = (mask > 0.5).float()

        encoded_img_256 = wip.encoder(image_256, message, mask_256[:, :1], jnd=True, jnd_factor=jnd_factor)
        encoded_img = (F.interpolate((encoded_img_256 - image_256), size=(H, W), mode="bilinear") + image).clamp_(-1, 1)
        encoded_img = encoded_img * mask_gt + image * (1 - mask_gt)

        # Metrics
        psnr = psnr(inverse_normalize(encoded_img), inverse_normalize(image), 1).item()
        ssim = torch.mean(ssim(inverse_normalize(encoded_img), inverse_normalize(image), window_size=11)).item()

        message_pred = wip.decoder(encoded_img_256)
        mask_pred = wip.segment(encoded_img_256)
        mask_pred = F.interpolate(mask_pred, size=(H, W), mode="bilinear")
        mask_pred = (mask_pred > 0.5).float()

        acc = bit_acc(message, message_pred)
        iou = iou_score(mask_gt, mask_pred)

        # Save outputs
        to_pil_image(mask_pred[0]).save(os.path.join(args.mask_pred_path, f"{base}.png"))
        to_pil_image((encoded_img[0] + 1) / 2).save(os.path.join(args.wm_image_path, f"{base}.png"))

        acc_sum += acc
        iou_sum += iou
        PSNR_sum += psnr
        SSIM_sum += ssim

    n = len(image_files)
    print(f"\nAverage Bit Acc: {acc_sum / n:.4f}")
    print(f"Average IoU: {iou_sum / n:.4f}")
    print(f"Average PSNR: {PSNR_sum / n:.4f}")
    print(f"Average SSIM: {SSIM_sum / n:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watermark Encoding & Evaluation")
    parser.add_argument("--config", type=str, default="configs/config_32.yaml",
                        help="Path to model config YAML")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/checkpoint_32.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--image_dir", type=str, default="images/examples",
                        help="Folder of input images")
    parser.add_argument("--mask_dir", type=str, required=False,
                        help="Folder of masks")
    parser.add_argument("--wm_image_path", type=str, default="results/watermarked",
                        help="Folder to save watermarked images")
    parser.add_argument("--msg_save_path", type=str, default="results/messages",
                        help="Folder to save embedded messages")
    parser.add_argument("--mask_pred_path", type=str, default="results/mask_preds",
                        help="Folder to save predicted masks")
    parser.add_argument("--jnd_factor", type=float, default=1.5,
                        help="JND factor for encoding")

    args = parser.parse_args()
    process_images(args)
