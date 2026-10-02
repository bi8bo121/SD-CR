import os
import sys
import math
import json
import time
import logging
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as skimage_ssim
from sklearn.metrics import mutual_info_score

import torch
import torch.nn.functional as F


DEFAULT_STAGE3_CKPT = 'stage3_results/stage3_ep027.pth'
DEFAULT_RESULTS_ROOT = 'test_results_stage3_ep027'
DEFAULT_OUTPUT_SUBDIR = 'fused_stage3_ep027'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('stage3_test')

@dataclass
class DatasetSpec:
    name: str
    root: str
    ir_dir: str
    vis_dir: str
    gt_dir: Optional[str] = None
    has_gt: bool = False


def load_dataset_specs(config_path: Path) -> List[DatasetSpec]:
    """Load dataset paths from JSON; relative roots use the JSON file's directory."""
    with config_path.open('r', encoding='utf-8') as f:
        document = json.load(f)
    if isinstance(document, dict):
        records = document.get('datasets', [document])
    else:
        records = document
    if not isinstance(records, list) or not records:
        raise ValueError('dataset_config must contain a nonempty datasets list')

    specs = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Each dataset_config entry must be an object')
        name = str(record['name'])
        if not name or name in ('.', '..') or Path(name).name != name:
            raise ValueError(f'Invalid dataset name: {name!r}')
        root = Path(record['root'])
        if not root.is_absolute():
            root = (config_path.parent / root).resolve()
        gt_dir = record.get('gt_dir')
        has_gt = bool(record.get('has_gt', gt_dir is not None))
        if has_gt and not gt_dir:
            raise ValueError(f'{name}: has_gt requires gt_dir')
        specs.append(DatasetSpec(name, str(root), str(record['ir_dir']),
                                 str(record['vis_dir']), gt_dir, has_gt))
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError('Dataset names in dataset_config must be unique')
    return specs


try:
    import cpbd 
    _CPBD_AVAILABLE = True
except Exception as e:
    cpbd = None
    _CPBD_AVAILABLE = False
    logger.warning(f'cpbd 导入失败，将把 CPBD 记为 NaN，不中断测试。错误: {e}')
try:
    import pyiqa
    _PYIQA_AVAILABLE = True
except Exception as e:
    pyiqa = None
    _PYIQA_AVAILABLE = False
    logger.warning(f'pyiqa 导入失败，将把 MUSIQ/BRISQUE 记为 NaN，不中断测试。错误: {e}')

def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def build_stem_map(files: List[Path]) -> Dict[str, Path]:
    return {p.stem.lower(): p for p in files}


def resolve_pairs(spec: DatasetSpec) -> List[Dict[str, Optional[Path]]]:
    root = Path(spec.root)
    ir_dir = root / spec.ir_dir
    vis_dir = root / spec.vis_dir
    gt_dir = root / spec.gt_dir if spec.gt_dir else None

    ir_files = list_images(ir_dir)
    vis_files = list_images(vis_dir)
    gt_files = list_images(gt_dir) if gt_dir else []

    if len(ir_files) == 0 or len(vis_files) == 0:
        logger.warning(f'[{spec.name}] 缺少输入图像，跳过。IR={ir_dir}, VIS={vis_dir}')
        return []

    ir_map = build_stem_map(ir_files)
    vis_map = build_stem_map(vis_files)

    if spec.has_gt:
        gt_map = build_stem_map(gt_files)
        common = sorted(set(ir_map.keys()) & set(vis_map.keys()) & set(gt_map.keys()))
    else:
        common = sorted(set(ir_map.keys()) & set(vis_map.keys()))

    pairs = []
    if common:
        for stem in common:
            pairs.append({
                'stem': stem,
                'ir': ir_map[stem],
                'vis': vis_map[stem],
                'gt': gt_map[stem] if spec.has_gt else None,
            })
        return pairs

    min_len = min(len(ir_files), len(vis_files), len(gt_files) if spec.has_gt else 10**9)
    logger.warning(f'[{spec.name}] stem 未对齐，回退到按排序配对，共 {min_len} 对。')
    for i in range(min_len):
        pairs.append({
            'stem': ir_files[i].stem.lower(),
            'ir': ir_files[i],
            'vis': vis_files[i],
            'gt': gt_files[i] if spec.has_gt else None,
        })
    return pairs


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img)


def load_gray(path: Path) -> np.ndarray:
    img = Image.open(path).convert('L')
    return np.array(img)


def ensure_same_hw(ref_hw: Tuple[int, int], img: np.ndarray, mode: str = 'rgb') -> np.ndarray:
    h, w = ref_hw
    if img.shape[0] == h and img.shape[1] == w:
        return img
    interp = cv2.INTER_LINEAR if mode == 'rgb' else cv2.INTER_LINEAR
    resized = cv2.resize(img, (w, h), interpolation=interp)
    logger.warning(f'发现尺寸不一致，已自动调整到 {(h, w)}')
    return resized


def np_rgb_to_tensor(img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img_rgb).float() / 255.0
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    return x


def np_gray_to_tensor(img_gray: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img_gray).float() / 255.0
    x = x.unsqueeze(0).unsqueeze(0).to(device)
    return x


def tensor_to_uint8_rgb(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu().clamp(0.0, 1.0)
    if x.dim() == 4:
        x = x[0]
    x = x.permute(1, 2, 0).numpy()
    x = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    return x


def rgb_to_gray_uint8(img_rgb: np.ndarray) -> np.ndarray:
    return np.round(cv2.cvtColor(img_rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)).astype(np.float32)


def pad_to_multiple(x: torch.Tensor, multiple: int = 8, mode: str = 'reflect') -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
    return x, (pad_h, pad_w)


def crop_back(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x


def compute_cpbd_safe(gray_uint8: np.ndarray) -> float:
    if not _CPBD_AVAILABLE:
        return float('nan')
    gray_uint8 = np.clip(gray_uint8, 0, 255).astype(np.uint8)
    for candidate in [gray_uint8, gray_uint8.astype(np.float32), gray_uint8.astype(np.float32) / 255.0]:
        try:
            return float(cpbd.compute(candidate))
        except Exception:
            pass
    return float('nan')

def build_pyiqa_metrics(device: torch.device, use_pyiqa: bool = False) -> Dict[str, object]:

    metrics = {}

    if not use_pyiqa:
        return metrics

    if not _PYIQA_AVAILABLE:
        logger.warning('pyiqa 不可用，MUSIQ/BRISQUE 将记为 NaN。')
        return metrics

    metric_names = {
        'MUSIQ': 'musiq',
        'BRISQUE': 'brisque',
    }

    for out_name, model_name in metric_names.items():
        try:
            metric = pyiqa.create_metric(model_name, device=device)
            if hasattr(metric, 'eval'):
                metric.eval()
            metrics[out_name] = metric

            lower_better = getattr(metric, 'lower_better', None)
            logger.info(f'已加载 IQA 指标: {out_name} / {model_name} | lower_better={lower_better}')

        except Exception as e:
            logger.warning(f'加载 IQA 指标 {out_name} 失败，将记为 NaN。错误: {e}')

    return metrics


def rgb_uint8_to_pyiqa_tensor(img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    pyiqa 输入格式:
    RGB, float32, [0, 1], shape = [N, 3, H, W]
    """
    img_rgb = np.clip(img_rgb, 0, 255).astype(np.float32) / 255.0
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    return x


@torch.inference_mode()
def compute_pyiqa_safe(
    fused_rgb: np.ndarray,
    iqa_metrics: Dict[str, object],
    device: torch.device,
) -> Dict[str, float]:
    out = {
        'MUSIQ': float('nan'),
        'BRISQUE': float('nan'),
    }

    if not iqa_metrics:
        return out

    try:
        x = rgb_uint8_to_pyiqa_tensor(fused_rgb, device)

        for name, metric in iqa_metrics.items():
            try:
                score = metric(x)

                if isinstance(score, (list, tuple)):
                    score = score[0]

                if isinstance(score, torch.Tensor):
                    score = score.detach().float().cpu().reshape(-1)[0].item()

                out[name] = float(score)

            except Exception as e:
                logger.warning(f'计算 {name} 失败，记为 NaN。错误: {e}')

    except Exception as e:
        logger.warning(f'构建 pyiqa 输入失败，MUSIQ/BRISQUE 记为 NaN。错误: {e}')

    return out


class Evaluator:
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None):
        if imgA is None:
            assert isinstance(imgF, np.ndarray), 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert isinstance(imgF, np.ndarray) and isinstance(imgA, np.ndarray) and isinstance(imgB, np.ndarray), 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()
        h = np.bincount(a, minlength=256).astype(np.float64)
        h = h / max(a.shape[0], 1)
        return float(-np.sum(h * np.log2(h + (h == 0))))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return float(np.std(img))

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return float(np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2)))

    @classmethod
    def AG(cls, img):
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)
        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return float(np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2)))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return float(mutual_info_score(image_F.flatten(), image_A.flatten()) +
                     mutual_info_score(image_F.flatten(), image_B.flatten()))

    @classmethod
    def MSE(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return float((np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2)

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        eps = 1e-12
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / (
            np.sqrt(np.sum((image_A - np.mean(image_A)) ** 2) * np.sum((image_F - np.mean(image_F)) ** 2)) + eps
        )
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / (
            np.sqrt(np.sum((image_B - np.mean(image_B)) ** 2) * np.sum((image_F - np.mean(image_F)) ** 2)) + eps
        )
        return float((rAF + rBF) / 2)

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        mse = cls.MSE(image_F, image_A, image_B)
        if mse <= 1e-12:
            return float('inf')
        maxv = float(np.max(image_F)) if np.max(image_F) > 0 else 255.0
        return float(10 * np.log10((maxv ** 2) / mse))

    @classmethod
    def SCD(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        eps = 1e-12
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / (
            np.sqrt(np.sum((image_A - np.mean(image_A)) ** 2) * np.sum((imgF_B - np.mean(imgF_B)) ** 2)) + eps
        )
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / (
            np.sqrt(np.sum((image_B - np.mean(image_B)) ** 2) * np.sum((imgF_A - np.mean(imgF_A)) ** 2)) + eps
        )
        return float(corr1 + corr2)

    @classmethod
    def compare_viff(cls, ref, dist):
        sigma_nsq = 2
        eps = 1e-10
        num = 0.0
        den = 0.0
        ref = ref.astype(np.float64)
        dist = dist.astype(np.float64)

        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            win = h / sumh if sumh != 0 else h

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den if den > eps else 1.0
        return 1.0 if np.isnan(vifp) else float(vifp)

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return float(cls.compare_viff(image_A, image_F) + cls.compare_viff(image_B, image_F))

    @classmethod
    def Qabf_getArray(cls, img):
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0] = np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls, aA, gA, aF, gF):
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        eps = 1e-12
        GAF = np.zeros_like(aA)
        QaAF = np.zeros_like(aA)
        QgAF = np.zeros_like(aA)

        mask1 = gA > gF
        mask2 = gA == gF
        mask3 = gA < gF
        GAF[mask1] = gF[mask1] / (gA[mask1] + eps)
        GAF[mask2] = gF[mask2]
        GAF[mask3] = gA[mask3] / (gF[mask3] + eps)
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        return QgAF * QaAF

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)
        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return float(nume / (deno + 1e-12))

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        s1 = skimage_ssim(image_F, image_A, data_range=255)
        s2 = skimage_ssim(image_F, image_B, data_range=255)
        return float(s1 + s2)


def compute_metrics_for_one(
    fused_gray: np.ndarray,
    ir_gray: np.ndarray,
    vis_gray: np.ndarray,
    has_gt: bool,
) -> Dict[str, float]:
    fused_gray = fused_gray.astype(np.float32)
    ir_gray = ir_gray.astype(np.float32)
    vis_gray = vis_gray.astype(np.float32)

    metrics = {
        'EN': Evaluator.EN(fused_gray),
        'SD': Evaluator.SD(fused_gray),
        'SF': Evaluator.SF(fused_gray),
        'AG': Evaluator.AG(fused_gray),
        'CPBD': compute_cpbd_safe(fused_gray),
    }

    if has_gt:
        metrics.update({
            'MI': Evaluator.MI(fused_gray, ir_gray, vis_gray),
            'MSE': Evaluator.MSE(fused_gray, ir_gray, vis_gray),
            'PSNR': Evaluator.PSNR(fused_gray, ir_gray, vis_gray),
            'CC': Evaluator.CC(fused_gray, ir_gray, vis_gray),
            'SSIM': Evaluator.SSIM(fused_gray, ir_gray, vis_gray),
            'SCD': Evaluator.SCD(fused_gray, ir_gray, vis_gray),
            'VIFF': Evaluator.VIFF(fused_gray, ir_gray, vis_gray),
            'Qabf': Evaluator.Qabf(fused_gray, ir_gray, vis_gray),
        })
    return metrics


def import_model_class():

    try:
        from models.sd_cro_stage2_lowdim import SDCRONet_Stage2_LowDim
        logger.info('models.sd_cro_stage2_lowdim.SDCRONet_Stage2_LowDim')
        return SDCRONet_Stage2_LowDim
    except Exception as e2:
        raise ImportError(f'无法导入测试模型类: {e2}')


def build_model(stage3_ckpt: str, stage1_ckpt: Optional[str], device: torch.device) -> torch.nn.Module:
    ModelClass = import_model_class()
    model = ModelClass(stage1_checkpoint_path=stage1_ckpt or None, encoder_channels=64)

    ckpt = torch.load(stage3_ckpt, map_location='cpu')
    state_dict = ckpt
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict', 'model_state_dict'):
            if key in ckpt and isinstance(ckpt[key], dict):
                state_dict = ckpt[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError('Stage3 checkpoint must contain a model state_dict')
    state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    return model


# ============================
# 推理
# ============================
@torch.inference_mode()
def forward_one(model: torch.nn.Module, ir_gray: np.ndarray, vis_rgb: np.ndarray, device: torch.device, pad_multiple: int = 8) -> np.ndarray:
    ir_t = np_gray_to_tensor(ir_gray, device)
    vis_t = np_rgb_to_tensor(vis_rgb, device)

    ir_t, pad_hw = pad_to_multiple(ir_t, multiple=pad_multiple, mode='reflect')
    vis_t, _ = pad_to_multiple(vis_t, multiple=pad_multiple, mode='reflect')

    out = model(ir_t, vis_t)
    if isinstance(out, dict):
        fused = out['fused_image']
    elif isinstance(out, (list, tuple)) and len(out) > 0:
        fused = out[0]
    else:
        raise TypeError('模型输出格式无法识别，期望 dict 且包含 fused_image。')

    fused = crop_back(fused, pad_hw)
    fused_rgb = tensor_to_uint8_rgb(fused)
    return fused_rgb


# ============================
# 保存与统计
# ============================
def nanmean_dict(records: List[Dict[str, float]], keys: List[str]) -> Dict[str, float]:
    out = {}
    for k in keys:
        vals = [r.get(k, np.nan) for r in records]
        vals = np.array(vals, dtype=np.float64)
        out[k] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float('nan')
    return out


def save_rgb(path: Path, img_rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_rgb).save(path)



def main():
    import argparse
    parser = argparse.ArgumentParser(description='Stage3 全数据集统一测试脚本')
    parser.add_argument('--stage3_ckpt', type=str, default=DEFAULT_STAGE3_CKPT)
    parser.add_argument('--stage1_ckpt', type=str, default=None,
                        help='可选的 Stage1 权重；完整 Stage3 权重单独推理不需要它')
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--dataset_root', type=str,
                               help='单数据集根目录，包含红外与可见光子目录')
    dataset_group.add_argument('--dataset_config', type=str,
                               help='JSON 数据集清单，含 datasets 数组')
    parser.add_argument('--dataset_name', type=str, default=None,
                        help='单数据集名称；默认取 dataset_root 的目录名')
    parser.add_argument('--ir_dir', type=str, default='ir', help='dataset_root 下的红外子目录')
    parser.add_argument('--vis_dir', type=str, default='vis', help='dataset_root 下的可见光子目录')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='可选的真值可见光子目录；指定后启用有真值指标')
    parser.add_argument('--results_root', type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument('--output_subdir', type=str, default=DEFAULT_OUTPUT_SUBDIR,
                        help='results_root 下保存融合图的相对子目录名')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--project_root', type=str, default='.', help='你的工程根目录，需包含 models/ 等代码')
    parser.add_argument('--pad_multiple', type=int, default=8)
    parser.add_argument('--only_gt', action='store_true')
    parser.add_argument('--only_no_gt', action='store_true')
    parser.add_argument('--skip_existing', action='store_true', help='若输出图已存在，则跳过推理但仍尝试读取并统计')

    parser.add_argument(
        '--use_pyiqa',
        action='store_true',
        help='启用 MUSIQ 和 BRISQUE 指标'
    )

    parser.add_argument(
        '--iqa_on_cpu',
        action='store_true',
        help='将 MUSIQ/BRISQUE 放到 CPU 上计算，节省 GPU 显存但速度较慢'
    )

    args = parser.parse_args()

    if args.only_gt and args.only_no_gt:
        raise ValueError('--only_gt 与 --only_no_gt 不能同时设置')
    output_subdir = Path(args.output_subdir)
    if output_subdir.is_absolute() or '..' in output_subdir.parts:
        parser.error('--output_subdir 必须是 results_root 下的相对目录')

    if args.dataset_config:
        dataset_list = load_dataset_specs(Path(args.dataset_config).resolve())
    else:
        dataset_root = Path(args.dataset_root).resolve()
        dataset_name = args.dataset_name or dataset_root.name
        if not dataset_name or dataset_name in ('.', '..') or Path(dataset_name).name != dataset_name:
            parser.error('--dataset_name 必须是单个目录名')
        dataset_list = [DatasetSpec(dataset_name, str(dataset_root),
                                    args.ir_dir, args.vis_dir, args.gt_dir,
                                    args.gt_dir is not None)]
    if args.only_gt:
        dataset_list = [spec for spec in dataset_list if spec.has_gt]
    elif args.only_no_gt:
        dataset_list = [spec for spec in dataset_list if not spec.has_gt]
    if not dataset_list:
        parser.error('筛选后没有待测试的数据集')

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        raise FileNotFoundError(f'project_root 不存在: {project_root}')
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    logger.info('开始构建模型...')
    model = build_model(args.stage3_ckpt, args.stage1_ckpt, device)
    logger.info(f'模型加载完成: {args.stage3_ckpt}')
    iqa_device = torch.device('cpu') if args.iqa_on_cpu else device
    iqa_metrics = build_pyiqa_metrics(iqa_device, use_pyiqa=args.use_pyiqa)

    all_detail_rows: List[Dict] = []
    all_summary_rows: List[Dict] = []

    for spec in dataset_list:
        pairs = resolve_pairs(spec)
        if len(pairs) == 0:
            continue

        logger.info('=' * 100)
        logger.info(f'测试数据集: {spec.name} | has_gt={spec.has_gt} | 样本数={len(pairs)}')
        logger.info('=' * 100)

        dataset_records: List[Dict[str, float]] = []
        save_dir = results_root / output_subdir / spec.name
        save_dir.mkdir(parents=True, exist_ok=True)

        pbar = tqdm(pairs, desc=f'{spec.name}', ncols=120)
        for item in pbar:
            stem = item['stem']
            ir_path = item['ir']
            vis_path = item['vis']
            gt_path = item['gt']
            save_path = save_dir / f'{spec.name}__{stem}.png'

            ir_gray = load_gray(ir_path)
            vis_rgb = load_rgb(vis_path)
            h0, w0 = ir_gray.shape[:2]
            vis_rgb = ensure_same_hw((h0, w0), vis_rgb, mode='rgb')

            if spec.has_gt and gt_path is not None:
                gt_rgb = load_rgb(gt_path)
                gt_rgb = ensure_same_hw((h0, w0), gt_rgb, mode='rgb')
                gt_gray = rgb_to_gray_uint8(gt_rgb)
            else:
                gt_gray = None

            if args.skip_existing and save_path.exists():
                fused_rgb = load_rgb(save_path)
            else:
                fused_rgb = forward_one(model, ir_gray, vis_rgb, device, pad_multiple=args.pad_multiple)
                save_rgb(save_path, fused_rgb)

            fused_gray = rgb_to_gray_uint8(fused_rgb)
            metrics = compute_metrics_for_one(
                fused_gray=fused_gray,
                ir_gray=ir_gray.astype(np.float32),
                vis_gray=(gt_gray if gt_gray is not None else vis_rgb[..., 0].astype(np.float32)),
                has_gt=spec.has_gt,
            )
            metrics.update(compute_pyiqa_safe(
                fused_rgb=fused_rgb,
                iqa_metrics=iqa_metrics,
                device=iqa_device,
            ))

            row = {
                'dataset': spec.name,
                'has_gt': spec.has_gt,
                'stem': stem,
                'ir_path': str(ir_path),
                'vis_path': str(vis_path),
                'gt_path': str(gt_path) if gt_path is not None else '',
                'save_path': str(save_path),
            }
            row.update(metrics)
            all_detail_rows.append(row)
            dataset_records.append(metrics)

            # shown_keys = ['EN', 'SD', 'SF', 'AG'] + (['PSNR', 'SSIM', 'Qabf'] if spec.has_gt else [])
            shown_keys = ['EN', 'SD', 'SF', 'AG', 'CPBD']

            if args.use_pyiqa:
                shown_keys += ['MUSIQ', 'BRISQUE']

            if spec.has_gt:
                shown_keys += ['PSNR', 'SSIM', 'Qabf']
            postfix = {k: f'{metrics[k]:.3f}' if np.isfinite(metrics[k]) else 'nan' for k in shown_keys if k in metrics}
            pbar.set_postfix(postfix)

        if spec.has_gt:
            metric_keys = [
                'EN', 'SD', 'SF', 'AG', 'CPBD', 'MUSIQ', 'BRISQUE',
                'MI', 'MSE', 'PSNR', 'CC', 'SSIM', 'SCD', 'VIFF', 'Qabf'
            ]
        else:
            metric_keys = ['EN', 'SD', 'SF', 'AG', 'CPBD', 'MUSIQ', 'BRISQUE']

        summary = nanmean_dict(dataset_records, metric_keys)
        summary_row = {
            'dataset': spec.name,
            'has_gt': spec.has_gt,
            'num_samples': len(dataset_records),
            'root': spec.root,
            'save_dir': str(save_dir),
        }
        summary_row.update(summary)
        all_summary_rows.append(summary_row)

        logger.info(f'[{spec.name}] 平均指标: ' + ' | '.join([
            f'{k}={summary_row[k]:.4f}' if np.isfinite(summary_row[k]) else f'{k}=nan'
            for k in metric_keys
        ]))

    detail_df = pd.DataFrame(all_detail_rows)
    summary_df = pd.DataFrame(all_summary_rows)

    detail_csv = results_root / 'all_details.csv'
    summary_csv = results_root / 'all_summary.csv'
    detail_xlsx = results_root / 'all_details.xlsx'
    summary_xlsx = results_root / 'all_summary.xlsx'
    config_json = results_root / 'run_config.json'

    detail_df.to_csv(detail_csv, index=False, encoding='utf-8-sig')
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')

    try:
        detail_df.to_excel(detail_xlsx, index=False)
        summary_df.to_excel(summary_xlsx, index=False)
    except Exception as e:
        logger.warning(f'导出 Excel 失败，仅保留 CSV。错误: {e}')

    with open(config_json, 'w', encoding='utf-8') as f:
        json.dump({
            'stage3_ckpt': args.stage3_ckpt,
            'stage1_ckpt': args.stage1_ckpt,
            'results_root': args.results_root,
            'output_subdir': args.output_subdir,
            'device': args.device,
            'project_root': str(project_root),
            'pad_multiple': args.pad_multiple,
            'cpbd_available': _CPBD_AVAILABLE,
            'datasets': [asdict(x) for x in dataset_list],
        }, f, ensure_ascii=False, indent=2)

    logger.info('=' * 100)
    logger.info('全部测试完成')
    logger.info(f'逐图结果: {detail_csv}')
    logger.info(f'数据集均值: {summary_csv}')
    logger.info(f'配置文件: {config_json}')
    logger.info('=' * 100)


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
