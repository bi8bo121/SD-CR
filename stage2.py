import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm

from models.sd_cro_stage2_lowdim import SDCRONet_Stage2_LowDim
from data.dataset_v2 import create_dataloaders
from utils.metrics_wrapper import FusionMetricsTracker
from trainers.fusion_trainer import FusionLoss


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Stage2Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        lambda_res: float = 0.5,
        samples_per_type: int = 5,
        save_dir: str = 'stage2_results_fresh'
    ):
        self.device = torch.device(device)
        self.lambda_res = lambda_res
        self.samples_per_type = samples_per_type

        self.raw_model = model.to(self.device)

        self.criterion_fusion = FusionLoss().to(self.device)
        self.criterion_restore = nn.L1Loss().to(self.device)

        self._setup_freeze_strategy()

        trainable_params = filter(lambda p: p.requires_grad, self.raw_model.parameters())
        self.optimizer = torch.optim.Adam(trainable_params, lr=learning_rate)

        if self.device.type == 'cuda' and self.device.index in (None, 0) and torch.cuda.device_count() > 1:
            logger.info(f"启用 DataParallel: {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.raw_model)
        else:
            self.model = self.raw_model

        self.tracker = FusionMetricsTracker(self.device)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

    def _setup_freeze_strategy(self):
        logger.info("配置参数冻结策略...")

        # 1) 先冻结全部
        for param in self.raw_model.parameters():
            param.requires_grad = False


        for param in self.raw_model.fusion_head.parameters():
            param.requires_grad = True
        logger.info(" -> [Unfrozen] Fusion Head")

        unfrozen_count = 0
        decoder = self.raw_model.restoration_net.decoder
        for name, param in decoder.named_parameters():
            if 'stage2' in name or 'tail' in name:
                param.requires_grad = True
                unfrozen_count += 1
        logger.info(f" -> [Unfrozen] Restoration Decoder Partial: {unfrozen_count} params")

    def train_epoch(self, loader, epoch):
        self.model.train()
        loss_stats = {'total': [], 'fus': [], 'res': []}

        loop = tqdm(loader, desc=f"Epoch {epoch} [Train]")
        for batch in loop:
            ir = batch['infrared_hq'].to(self.device, non_blocking=True)
            vis_lq = batch['visible_lq'].to(self.device, non_blocking=True)
            vis_hq = batch['visible_hq'].to(self.device, non_blocking=True) 
            out = self.model(ir, vis_lq)

            loss_fus = self.criterion_fusion(out['fused_image'], ir, vis_hq)
            loss_res = self.criterion_restore(out['restored_image'], vis_hq)
            loss_total = loss_fus + self.lambda_res * loss_res

            self.optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            self.optimizer.step()

            loss_stats['total'].append(loss_total.item())
            loss_stats['fus'].append(loss_fus.item())
            loss_stats['res'].append(loss_res.item())

            loop.set_postfix(
                loss=f"{loss_total.item():.4f}",
                fus=f"{loss_fus.item():.4f}",
                res=f"{loss_res.item():.4f}"
            )

        return {k: float(np.mean(v)) for k, v in loss_stats.items()}

    @torch.no_grad()
    def validate(self, loader, epoch, save_images=False):

        self.model.eval()
        self.tracker.reset()
        vis_samples = {}

        loop = tqdm(loader, desc=f"Epoch {epoch} [Val]")
        for batch in loop:
            ir = batch['infrared_hq'].to(self.device, non_blocking=True)
            vis_lq = batch['visible_lq'].to(self.device, non_blocking=True)
            deg_types = batch['degradation_type']

            out = self.model(ir, vis_lq)
            fused = out['fused_image']
            self.tracker.update(fused)

            if save_images:
                for i in range(len(ir)):
                    full_name = str(deg_types[i])
                    parts = full_name.split('_')
                    dtype = "_".join(parts[:-1]) if len(parts) > 1 else full_name

                    if dtype not in vis_samples:
                        vis_samples[dtype] = []

                    if len(vis_samples[dtype]) < self.samples_per_type:
                        vis_samples[dtype].append({
                            'ir': ir[i].cpu(),
                            'vis': vis_lq[i].cpu(),
                            'res': out['restored_image'][i].cpu(),
                            'fus': fused[i].cpu()
                        })

        avg_metrics = self.tracker.compute()
        if save_images and len(vis_samples) > 0:
            self._save_vis(vis_samples, epoch)
        return avg_metrics

    def _save_vis(self, samples_dict, epoch):
        save_path = self.save_dir / f"epoch_{epoch:03d}"
        save_path.mkdir(exist_ok=True, parents=True)

        for dtype, samples_list in samples_dict.items():
            dtype_folder = save_path / dtype
            dtype_folder.mkdir(exist_ok=True, parents=True)

            for idx, imgs in enumerate(samples_list):
                ir = imgs['ir']
                ir_3c = ir.repeat(3, 1, 1) if ir.shape[0] == 1 else ir
                row = torch.cat([ir_3c, imgs['vis'], imgs['res'], imgs['fus']], dim=2)

                file_name = f"{dtype}_{idx:02d}.png"
                vutils.save_image(
                    row,
                    dtype_folder / file_name,
                    normalize=True,
                    value_range=(0, 1)
                )
            logger.info(f"保存 {dtype}: {len(samples_list)} 张图片")


def main():
    parser = argparse.ArgumentParser(description='Train SD-CR Stage 2 fusion model')
    parser.add_argument('--data-root', default='datasets/DDL', help='Root of prepared DDL data')
    parser.add_argument('--stage1-ckpt', default='checkpoints_l1_balanced_5types/restoration_epoch_150.pt')
    parser.add_argument('--save-dir', default='stage2_results_fresh')
    parser.add_argument('--resume', default=None, help='Stage 2 full checkpoint for explicit resumption')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    stage1_ckpt = args.stage1_ckpt
    resume_ckpt = args.resume

    num_epochs = 150
    vis_interval = 5

    logger.info("初始化 Stage 2 模型（加载 Stage1 150epoch 权重）...")
    model = SDCRONet_Stage2_LowDim(
        stage1_checkpoint_path=stage1_ckpt,
        encoder_channels=64
    )

    train_loader, val_loader, _ = create_dataloaders(
        root_dir=args.data_root,
        batch_size=10,
        val_batch_size=1,
        num_workers=2
    )

    trainer = Stage2Trainer(
        model=model,
        device=args.device,
        learning_rate=1e-4,
        lambda_res=0.5,
        samples_per_type=5,
        save_dir=args.save_dir
    )

    start_epoch = 1

    if resume_ckpt is not None:
        if not Path(resume_ckpt).is_file():
            raise FileNotFoundError(f"找不到 Stage 2 续训断点: {resume_ckpt}")
        logger.info(f"检测到 Stage2 断点，准备恢复训练: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location='cpu')

        trainer.raw_model.load_state_dict(ckpt['model'], strict=True)
        trainer.optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1

        logger.info(f"已恢复到 epoch {ckpt['epoch']}，将从 epoch {start_epoch} 继续训练")
    else:
        logger.info("未指定续训断点，开始从 epoch 1 训练 Stage2...")


    for epoch in range(start_epoch, num_epochs + 1):
        loss_dict = trainer.train_epoch(train_loader, epoch)

        save_images_this_epoch = (epoch % vis_interval == 0)
        metrics = trainer.validate(val_loader, epoch, save_images=save_images_this_epoch)

        log_str = (
            f"Ep {epoch} | "
            f"Total: {loss_dict['total']:.4f} | "
            f"Fus: {loss_dict['fus']:.4f} | "
            f"Res: {loss_dict['res']:.4f} | "
        )
        metric_str = " | ".join([f"{k}:{v:.3f}" for k, v in metrics.items()])
        logger.info(log_str + metric_str)

        save_path = trainer.save_dir / f"stage2_ep{epoch:03d}.pth"
        torch.save(trainer.raw_model.state_dict(), save_path)
        logger.info(f"已保存模型: {save_path}")

        full_ckpt = trainer.save_dir / f"stage2_ep{epoch:03d}_full.pth"
        torch.save({
            'epoch': epoch,
            'model': trainer.raw_model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }, full_ckpt)
        logger.info(f"已保存完整断点: {full_ckpt}")


if __name__ == '__main__':
    main()
