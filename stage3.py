import argparse
import os
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from tqdm import tqdm

from models.sd_cro_stage2_lowdim import SDCRONet_Stage2_LowDim
from data.dataset_v2 import create_dataloaders
from utils.metrics_wrapper import FusionMetricsTracker
from trainers.fusion_trainer import FusionLoss

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class Stage3Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',

        num_epochs: int = 30,
        phase1_epochs: int = 8,
        lambda_res: float = 2.0,

        lr_p1_fusion: float = 1e-5,
        lr_p1_decoder: float = 8e-6,
        lr_p1_reason: float = 8e-6,

        lr_p2_backbone: float = 2e-6,
        lr_p2_reason: float = 5e-6,
        lr_p2_decoder: float = 8e-6,
        lr_p2_fusion: float = 1e-5,

        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,

        samples_per_type: int = 5,
        save_dir: str = 'stage3_results-zuiz'
    ):
        self.device = torch.device(device)

        self.num_epochs = num_epochs
        self.phase1_epochs = phase1_epochs
        self.lambda_res = lambda_res

        self.lr_p1_fusion = lr_p1_fusion
        self.lr_p1_decoder = lr_p1_decoder
        self.lr_p1_reason = lr_p1_reason

        self.lr_p2_backbone = lr_p2_backbone
        self.lr_p2_reason = lr_p2_reason
        self.lr_p2_decoder = lr_p2_decoder
        self.lr_p2_fusion = lr_p2_fusion

        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.samples_per_type = samples_per_type

        self.raw_model = model.to(self.device)

        self.criterion_restore = nn.L1Loss().to(self.device)
        self.criterion_fusion = FusionLoss().to(self.device)

        self.tracker = FusionMetricsTracker(self.device)

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.current_phase = None
        self.optimizer = None
        self.scheduler = None

        self._configure_phase(phase=1)

        if self.device.type == 'cuda' and self.device.index in (None, 0) and torch.cuda.device_count() > 1:
            logger.info(f"start DataParallel: {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.raw_model)
        else:
            self.model = self.raw_model

    def _setup_phase1_freeze(self):
        """
        Phase 1 只微调：
        - compress_res / compress_ir / fusion_head
        - restoration_net.decoder
        - restoration_net.structure_domain
        - restoration_net.degradation_domain
        - restoration_net.conflict_map
        - restoration_net.sd_cro
        - restoration_net.restoration_net

        先冻结：
        - restoration_net.encoder_ir
        - restoration_net.encoder_vis
        """
        logger.info("Stage3 Phase 1: 先微调恢复相关链路 + 融合头")

        phase1_prefixes = [
            'compress_res',
            'compress_ir',
            'fusion_head',

            'restoration_net.decoder',
            'restoration_net.structure_domain',
            'restoration_net.degradation_domain',
            'restoration_net.conflict_map',
            'restoration_net.sd_cro',
            'restoration_net.restoration_net',
        ]

        total_params = 0
        trainable_params = 0
        block_counter = {k: 0 for k in phase1_prefixes}

        for name, param in self.raw_model.named_parameters():
            total_params += param.numel()

            train_flag = False
            for prefix in phase1_prefixes:
                if name.startswith(prefix):
                    train_flag = True
                    block_counter[prefix] += param.numel()
                    break

            param.requires_grad = train_flag
            if train_flag:
                trainable_params += param.numel()

        logger.info(f" -> Total params: {total_params:,}")
        logger.info(f" -> Trainable params (Phase 1): {trainable_params:,}")
        for k, v in block_counter.items():
            logger.info(f"    [{k}] {v:,}")

    def _setup_phase2_unfreeze(self):

        logger.info("Stage3 Phase 2: 全网小学习率微调")
        total_params = 0
        for _, param in self.raw_model.named_parameters():
            param.requires_grad = True
            total_params += param.numel()
        logger.info(f" -> All params trainable: {total_params:,}")

    def _split_trainable_params(self):

        fusion_params = []
        decoder_params = []
        reason_params = []
        backbone_params = []

        visited = set()

        for name, param in self.raw_model.named_parameters():
            if not param.requires_grad:
                continue

            pid = id(param)
            if pid in visited:
                continue
            visited.add(pid)

            if (
                name.startswith('compress_res') or
                name.startswith('compress_ir') or
                name.startswith('fusion_head')
            ):
                fusion_params.append(param)

            elif name.startswith('restoration_net.decoder'):
                decoder_params.append(param)

            elif (
                name.startswith('restoration_net.structure_domain') or
                name.startswith('restoration_net.degradation_domain') or
                name.startswith('restoration_net.conflict_map') or
                name.startswith('restoration_net.sd_cro') or
                name.startswith('restoration_net.restoration_net')
            ):
                reason_params.append(param)

            else:
                backbone_params.append(param)

        return fusion_params, decoder_params, reason_params, backbone_params

    def _build_optimizer(self):
        fusion_params, decoder_params, reason_params, backbone_params = self._split_trainable_params()
        param_groups = []

        if self.current_phase == 1:
            if fusion_params:
                param_groups.append({'params': fusion_params, 'lr': self.lr_p1_fusion})
            if decoder_params:
                param_groups.append({'params': decoder_params, 'lr': self.lr_p1_decoder})
            if reason_params:
                param_groups.append({'params': reason_params, 'lr': self.lr_p1_reason})
        else:
            if backbone_params:
                param_groups.append({'params': backbone_params, 'lr': self.lr_p2_backbone})
            if reason_params:
                param_groups.append({'params': reason_params, 'lr': self.lr_p2_reason})
            if decoder_params:
                param_groups.append({'params': decoder_params, 'lr': self.lr_p2_decoder})
            if fusion_params:
                param_groups.append({'params': fusion_params, 'lr': self.lr_p2_fusion})

        if len(param_groups) == 0:
            raise RuntimeError("没有可训练参数，请检查模块名匹配。")

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.99),
            weight_decay=self.weight_decay
        )

        logger.info("Optimizer param groups:")
        for i, group in enumerate(self.optimizer.param_groups):
            logger.info(f" -> Group {i}: lr={group['lr']:.2e}, params={len(group['params'])}")

    def _build_scheduler(self):
        if self.current_phase == 1:
            t_max = max(1, self.phase1_epochs)
            eta_min = 1e-6
        else:
            t_max = max(1, self.num_epochs - self.phase1_epochs)
            eta_min = 1e-7

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=t_max,
            eta_min=eta_min
        )

        logger.info(
            f"Scheduler built: phase={self.current_phase}, "
            f"T_max={t_max}, eta_min={eta_min:.1e}"
        )

    def _configure_phase(self, phase: int):
        self.current_phase = phase

        if phase == 1:
            self._setup_phase1_freeze()
        elif phase == 2:
            self._setup_phase2_unfreeze()
        else:
            raise ValueError(f"Unsupported phase: {phase}")

        self._build_optimizer()
        self._build_scheduler()

    def _maybe_switch_phase(self, epoch: int):
        if self.current_phase == 1 and epoch == self.phase1_epochs + 1:
            logger.info("=" * 80)
            logger.info(f"Epoch {epoch}: 切换到 Phase 2（全网微调）")
            logger.info("=" * 80)
            self._configure_phase(phase=2)

    def _trainable_params(self):
        return [p for p in self.raw_model.parameters() if p.requires_grad]

    def _get_current_lr_str(self) -> str:
        return ",".join([f"{pg['lr']:.2e}" for pg in self.optimizer.param_groups])

    def train_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self._maybe_switch_phase(epoch)

        self.model.train()
        stats = {'total': [], 'fus': [], 'res': []}

        loop = tqdm(loader, desc=f"Epoch {epoch} [Train][P{self.current_phase}]")
        for batch in loop:
            ir = batch['infrared_hq'].to(self.device, non_blocking=True)
            vis_lq = batch['visible_lq'].to(self.device, non_blocking=True)
            vis_hq = batch['visible_hq'].to(self.device, non_blocking=True)

            out = self.model(ir, vis_lq)
            fused = out['fused_image']
            restored = out['restored_image']

            loss_fus = self.criterion_fusion(fused, ir, vis_hq)
            loss_res = self.criterion_restore(restored, vis_hq)
            loss_total = loss_fus + self.lambda_res * loss_res

            self.optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            nn.utils.clip_grad_norm_(self._trainable_params(), max_norm=self.grad_clip)
            self.optimizer.step()

            stats['total'].append(loss_total.item())
            stats['fus'].append(loss_fus.item())
            stats['res'].append(loss_res.item())

            loop.set_postfix(
                phase=self.current_phase,
                total=f"{loss_total.item():.4f}",
                fus=f"{loss_fus.item():.4f}",
                res_l1=f"{loss_res.item():.4f}"
            )

        return {k: float(np.mean(v)) for k, v in stats.items()}

    @torch.no_grad()
    def validate_and_save_visuals(self, loader, epoch: int) -> Dict[str, float]:

        self.model.eval()
        self.tracker.reset()

        vis_samples = {}
        save_counters = {}

        loop = tqdm(loader, desc=f"Epoch {epoch} [ValSave][P{self.current_phase}]")
        for batch in loop:
            ir = batch['infrared_hq'].to(self.device, non_blocking=True)
            vis_lq = batch['visible_lq'].to(self.device, non_blocking=True)
            deg_types = batch['degradation_type']

            out = self.model(ir, vis_lq)
            fused = out['fused_image'].clamp(0, 1)
            restored = out['restored_image'].clamp(0, 1)

            try:
                self.tracker.update(fused)
            except Exception:
                pass

            for i in range(len(ir)):
                full_name = str(deg_types[i])
                parts = full_name.split('_')
                dtype = "_".join(parts[:-1]) if len(parts) > 1 else full_name

                if dtype not in vis_samples:
                    vis_samples[dtype] = []
                    save_counters[dtype] = 0

                can_save = (self.samples_per_type < 0) or (save_counters[dtype] < self.samples_per_type)
                if can_save:
                    vis_samples[dtype].append({
                        'ir': ir[i].detach().cpu(),
                        'vis': vis_lq[i].detach().cpu(),
                        'res': restored[i].detach().cpu(),
                        'fus': fused[i].detach().cpu(),
                    })
                    save_counters[dtype] += 1

        metrics = {}
        try:
            metrics = self.tracker.compute()
        except Exception:
            metrics = {}

        if len(vis_samples) > 0:
            self._save_vis(vis_samples, epoch)

        return metrics

    def _save_vis(self, samples_dict, epoch: int):
        save_path = self.save_dir / f"epoch_{epoch:03d}"
        save_path.mkdir(parents=True, exist_ok=True)

        for dtype, sample_list in samples_dict.items():
            dtype_folder = save_path / dtype
            dtype_folder.mkdir(parents=True, exist_ok=True)

            for idx, imgs in enumerate(sample_list):
                ir = imgs['ir']
                vis = imgs['vis']
                res = imgs['res']
                fus = imgs['fus']

                ir_3c = ir.repeat(3, 1, 1) if ir.shape[0] == 1 else ir
                row = torch.cat([ir_3c, vis, res, fus], dim=2)

                file_name = f"{dtype}_{idx:04d}.png"
                vutils.save_image(
                    row,
                    dtype_folder / file_name,
                    normalize=True,
                    value_range=(0, 1)
                )

        logger.info(f"已保存可视化结果到: {save_path}")

    def save_checkpoint(self, epoch: int):
        model_ckpt = self.save_dir / f"stage3_ep{epoch:03d}.pth"
        full_ckpt = self.save_dir / f"stage3_ep{epoch:03d}_full.pth"

        torch.save(self.raw_model.state_dict(), model_ckpt)
        torch.save({
            'epoch': epoch,
            'phase': self.current_phase,
            'model': self.raw_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
        }, full_ckpt)

        logger.info(f"已保存模型: {model_ckpt}")
        logger.info(f"已保存完整断点: {full_ckpt}")

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()


def load_stage2_weights(model: nn.Module, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 Stage2 权重: {ckpt_path}")

    logger.info(f"加载 Stage2 权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu')

    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt

    # 兼容 DataParallel 保存
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    logger.info("Stage2 权重加载成功！")


def main():
    parser = argparse.ArgumentParser(description='Train SD-CR Stage 3 fine-tuning model')
    parser.add_argument('--data-root', default='datasets/DDL', help='Root of prepared DDL data')
    parser.add_argument('--stage1-ckpt', default='checkpoints_l1_balanced_5types/restoration_epoch_150.pt')
    parser.add_argument('--stage2-ckpt', default='stage2_results_fresh/stage2_ep120_full.pth')
    parser.add_argument('--save-dir', default='stage3_results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    stage1_ckpt = args.stage1_ckpt
    stage2_best_ckpt = args.stage2_ckpt
    root_dir = args.data_root
    batch_size = 10       
    val_batch_size = 5   
    num_workers = 2

    num_epochs = 30
    phase1_epochs = 8
    lambda_res = 2.0


    samples_per_type = 5

    logger.info("=" * 80)
    logger.info("初始化 Stage3 模型（基于 Stage2 LowDim 继续微调）")
    logger.info("=" * 80)

    model = SDCRONet_Stage2_LowDim(
        stage1_checkpoint_path=stage1_ckpt,
        encoder_channels=64
    )

    load_stage2_weights(model, stage2_best_ckpt)

    train_loader, val_loader, _ = create_dataloaders(
        root_dir=root_dir,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers
    )

    trainer = Stage3Trainer(
        model=model,
        device=args.device,
        num_epochs=num_epochs,
        phase1_epochs=phase1_epochs,
        lambda_res=lambda_res,

        lr_p1_fusion=1e-5,
        lr_p1_decoder=8e-6,
        lr_p1_reason=8e-6,

        lr_p2_backbone=2e-6,
        lr_p2_reason=5e-6,
        lr_p2_decoder=8e-6,
        lr_p2_fusion=1e-5,

        weight_decay=1e-4,
        grad_clip=1.0,
        samples_per_type=samples_per_type,
        save_dir=args.save_dir
    )

    logger.info("=" * 80)
    logger.info("开始 Stage3 训练")
    logger.info("=" * 80)

    for epoch in range(1, num_epochs + 1):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_metrics = trainer.validate_and_save_visuals(val_loader, epoch)

        lr_str = trainer._get_current_lr_str()
        log_str = (
            f"Epoch {epoch:03d} | "
            f"Phase {trainer.current_phase} | "
            f"LR [{lr_str}] | "
            f"Train Total: {train_loss['total']:.4f} | "
            f"Train Fus: {train_loss['fus']:.4f} | "
            f"Train Res(L1): {train_loss['res']:.4f}"
        )

        if isinstance(val_metrics, dict) and len(val_metrics) > 0:
            extra = []
            for k, v in val_metrics.items():
                try:
                    extra.append(f"{k}:{v:.3f}")
                except Exception:
                    pass
            if len(extra) > 0:
                log_str += " | " + " | ".join(extra)

        logger.info(log_str)

        trainer.save_checkpoint(epoch)
        trainer.step_scheduler()

    logger.info("=" * 80)
    logger.info("Stage3 训练完成！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
