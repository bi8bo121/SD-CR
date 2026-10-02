import argparse
import piq
import torch
import torch.nn as nn
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict
import torchvision.utils as vutils

from models.sd_cro_resnet import SDCROResNet
from losses.stage_specific_losses_v2 import JointRestorationLoss_v2
from data.dataset_v2 import create_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BALANCED_TRAIN_TYPE_BATCH_MAP = {
    "VI_Blur": 2,
    "VI_Haze": 2,
    "VI_Low_light": 2,
    "VI_Noise": 2,
    "VI_Over_exposure": 2
}


class NoRefMetricsCalculator:
    def __init__(self, device='cuda'):
        self.device = device

    @torch.no_grad()
    def compute_metrics(self, restored_image: torch.Tensor) -> Dict[str, float]:
        restored_image = torch.clamp(restored_image, 0.0, 1.0)
        try:
            brisque_score = piq.brisque(
                restored_image,
                data_range=1.0,
                reduction='mean'
            ).item()
        except Exception:
            brisque_score = 0.0
        return {'brisque': brisque_score}


class RestorationOnlyTrainer:
    def __init__(
        self,
        restoration_model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 5e-5,
        logger_obj=None,
        val_vis_dir='val_results_l1_balanced_5types'
    ):
        self.restoration_model = restoration_model.to(device)
        self.device = torch.device(device)
        self.logger = logger_obj or logging.getLogger(__name__)

        restoration_params = list(self.restoration_model.parameters())
        self.logger.info(f"恢复网络参数量: {sum(p.numel() for p in restoration_params):,}")

        self.optimizer = torch.optim.AdamW(
            restoration_params,
            lr=learning_rate,
            weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=150,
            eta_min=1e-6
        )

        self.restoration_loss = JointRestorationLoss_v2(weight_l1=1.0).to(device)
        self.noref_metrics_calc = NoRefMetricsCalculator(device=device)

        self.val_vis_dir = Path(val_vis_dir)
        self.val_vis_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        self.restoration_model.train()
        total_loss = 0.0
        loss_stats = {}

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in progress_bar:
            infrared = batch['infrared_hq'].to(self.device, non_blocking=True)
            visible_lq = batch['visible_lq'].to(self.device, non_blocking=True)
            visible_gt = batch['visible_hq'].to(self.device, non_blocking=True)

            outputs = self.restoration_model(infrared, visible_lq)
            restored_image = outputs['restored_image']
            struct_atoms = outputs['struct_atoms']
            degradation_atoms = outputs['degradation_atoms']

            loss_total, loss_dict = self.restoration_loss(
                pred=restored_image,
                target=visible_gt,
                struct_atoms=struct_atoms,
                degradation_atoms=degradation_atoms
            )

            self.optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(self.restoration_model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss_total.item()
            for key, value in loss_dict.items():
                loss_stats.setdefault(key, []).append(value)

            progress_bar.set_postfix({
                'loss': f"{loss_total.item():.4f}",
                'l1': f"{loss_dict.get('l1_loss', 0):.4f}",
            })

        avg_loss_stats = {k: float(np.mean(v)) for k, v in loss_stats.items()}
        avg_loss_stats['loss'] = total_loss / len(train_loader)
        return avg_loss_stats

    @torch.no_grad()
    def validate(self, val_loader, epoch: int = None) -> Dict[str, float]:
        self.restoration_model.eval()
        metric_stats = {'brisque': []}

        epoch_dir = self.val_vis_dir / f'epoch_{epoch:03d}'
        epoch_dir.mkdir(parents=True, exist_ok=True)

        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            infrared = batch['infrared_hq'].to(self.device, non_blocking=True)
            visible_lq = batch['visible_lq'].to(self.device, non_blocking=True)
            deg_types = batch['degradation_type']

            outputs = self.restoration_model(infrared, visible_lq)
            restored_image = outputs['restored_image']

            batch_metrics = self.noref_metrics_calc.compute_metrics(restored_image)
            metric_stats['brisque'].append(batch_metrics.get('brisque', 0.0))

            if batch_idx < 3:
                for i in range(len(visible_lq)):
                    full_type_name = deg_types[i]
                    if "_" in full_type_name:
                        idx = full_type_name.rfind("_")
                        base_type = full_type_name[:idx]
                        intensity = full_type_name[idx + 1:]
                    else:
                        base_type = full_type_name
                        intensity = "def"

                    type_dir = epoch_dir / base_type
                    type_dir.mkdir(exist_ok=True)

                    img_lq = visible_lq[i].detach().cpu()
                    img_res = restored_image[i].detach().cpu()

                    comparison = torch.cat([img_lq, img_res], dim=2)
                    save_name = f"{intensity}_b{batch_idx}_{i}.png"
                    vutils.save_image(
                        comparison,
                        type_dir / save_name,
                        normalize=False
                    )

        avg_metrics = {k: float(np.mean(v)) for k, v in metric_stats.items()}
        self.logger.info(f"Val Epoch {epoch} | BRISQUE (↓): {avg_metrics.get('brisque', 0):.4f}")
        return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Train SD-CR Stage 1 restoration model')
    parser.add_argument('--data-root', default='datasets/DDL', help='Root of prepared DDL data')
    parser.add_argument('--checkpoint-dir', default='checkpoints_l1_balanced_5types')
    parser.add_argument('--val-vis-dir', default='val_results_l1_balanced_5types')
    parser.add_argument('--resume', default=None, help='Stage 1 full checkpoint for explicit resumption')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    num_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    logger.info(f"检测到 {num_gpus} 张GPU")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 80)
    logger.info("SD-CRO ResNet Stage1")
    logger.info("=" * 80)

    logger.info("Creating model...")
    restoration_model = SDCROResNet(
        encoder_channels=64,
        encoder_blocks=[2, 2, 4],
        decoder_blocks=[2, 2],
        num_atoms=16,
        degradation_dim=256
    )

    if num_gpus > 1 and device.index in (None, 0):
        logger.info(f"正在启动多卡并行训练，GPU数量: {num_gpus}")
        restoration_model = nn.DataParallel(restoration_model)
    else:
        logger.info("使用单卡训练")

    total_params = sum(p.numel() for p in restoration_model.parameters())
    logger.info(f"Model Params: {total_params / 1e6:.2f} M")

    trainer = RestorationOnlyTrainer(
        restoration_model=restoration_model,
        device=device,
        learning_rate=5e-5,
        logger_obj=logger,
        val_vis_dir=args.val_vis_dir
    )

    logger.info("\nLoading data...")
    current_batch_size = 10
    val_batch_size = 5
    num_epochs = 150

    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=args.data_root,
        batch_size=current_batch_size,
        val_batch_size=val_batch_size,
        num_workers=4,
        target_size=(320, 320),
        train_type_batch_map=BALANCED_TRAIN_TYPE_BATCH_MAP
    )

    start_epoch = 1

    resume_ckpt = args.resume
    if resume_ckpt is not None:
        if not Path(resume_ckpt).is_file():
            raise FileNotFoundError(f"找不到 Stage 1 续训断点: {resume_ckpt}")
        logger.info(f"加载断点: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)

        state_dict = ckpt['restoration_state_dict']

        model_is_dp = isinstance(trainer.restoration_model, nn.DataParallel)
        ckpt_has_module = next(iter(state_dict.keys())).startswith('module.')

        if model_is_dp and not ckpt_has_module:
            state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        elif (not model_is_dp) and ckpt_has_module:
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        trainer.restoration_model.load_state_dict(state_dict, strict=True)
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = int(ckpt['epoch']) + 1
        logger.info(f"断点续训成功，从 Epoch {start_epoch} 开始")
    else:
        logger.info("未指定续训断点，将从 epoch 1 开始训练")

    logger.info("\n" + "=" * 80)
    logger.info(f"Starting Training Loop (Batch Size: {current_batch_size})")
    logger.info(f"Fixed train_type_batch_map = {BALANCED_TRAIN_TYPE_BATCH_MAP}")
    logger.info(f"Epoch range: {start_epoch} -> {num_epochs}")
    logger.info("=" * 80 + "\n")

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)

        logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss.get('loss', 0):.6f} | "
            f"L1: {train_loss.get('l1_loss', 0):.6f} | "
            f"LR: {trainer.scheduler.get_last_lr()[0]:.2e}"
        )

        if epoch % 5 == 0:
            val_metrics = trainer.validate(val_loader, epoch=epoch)

            if isinstance(trainer.restoration_model, nn.DataParallel):
                restoration_state_dict = trainer.restoration_model.module.state_dict()
            else:
                restoration_state_dict = trainer.restoration_model.state_dict()

            ckpt = {
                'epoch': epoch,
                'restoration_state_dict': restoration_state_dict,
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'scheduler_state_dict': trainer.scheduler.state_dict(),
                'train_loss': train_loss.get('loss', 0),
                'val_brisque': val_metrics.get('brisque', 0)
            }
            ckpt_path = checkpoint_dir / f'restoration_epoch_{epoch:03d}.pt'
            torch.save(ckpt, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}\n")

        trainer.scheduler.step()

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
