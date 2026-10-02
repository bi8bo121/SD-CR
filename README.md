# SD-CR

**Structure–Degradation Co-Reasoning for Infrared–Visible Image Fusion under Asymmetric Degradation**

本项目包含三阶段训练代码，以及用于图像融合推理和评估的 `test.py`。Stage 1 训练可见光恢复网络；Stage 2 在 Stage 1 权重基础上训练融合分支；Stage 3 从 Stage 2 权重继续微调。模型结构、损失和训练数值参数沿用原实现。

## 代码结构

```text
├── stage1.py           # Stage 1：恢复网络训练
├── stage2.py           # Stage 2：融合网络训练
├── stage3.py           # Stage 3：分阶段微调
├── test.py             # Stage 3 推理及融合指标评估
├── models/             # 模型及其依赖模块
├── data/               # 数据读取与训练增强代码
├── losses/             # Stage 1 损失
├── trainers/           # Stage 2/3 融合损失
├── utils/              # Stage 2/3 训练验证指标
├── configs/            # 数据集清单示例
├── requirements.txt
└── .gitignore
```

以下命令均从项目根目录执行。

## 环境安装

代码需要 Python 3.9 或更新版本。建议先建立独立环境，并安装与本机 CPU/CUDA 平台匹配的 PyTorch 和 torchvision：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` 按代码导入列出依赖包，没有指定未经确认的精确版本。Stage 1 使用 `piq.brisque`，因此需要 `piq`。`test.py` 的 CPBD 指标依赖可选包 `cpbd`；传入 `--use_pyiqa` 时，MUSIQ/BRISQUE 指标还需要 `pyiqa`；Excel 输出需要 `openpyxl`。缺少这些可选包时，相关指标可能记为 NaN，或只生成 CSV。

## 数据准备

`data/dataset_v2.py` 读取**预先准备好的退化图像**，项目内没有完整的退化生成流水线。训练数据包含五类退化：`VI_Blur`、`VI_Haze`、`VI_Low_light`、`VI_Noise`、`VI_Over_exposure`；每类使用 `slight`、`moderate`、`average`、`extreme` 四种强度。目录示例：

```text
datasets/DDL/
└── VI_Blur/
    └── VI_Blur_slight/
        ├── train/
        │   ├── Infrared/    # 灰度红外图
        │   ├── Visible/     # 退化 RGB 可见光图
        │   └── Visible_gt/  # 干净 RGB 可见光图
        └── test/
            ├── Infrared/
            └── Visible/
```

其他退化类型和强度采用相同结构。同一组的 IR、VIS、GT 图像需使用相同文件主名。代码也接受 `VI_Blur/slight/{train,test}/...` 这种目录形式。训练时图像缩放到 320×320，并随机裁成 256×256；训练过程中的验证数据来自 `test/`，该划分没有 `Visible_gt`。

## 权重准备

本仓库不包含数据集或模型权重。默认路径及用途如下，也可以在命令中传入实际存放路径：

| 文件 | 默认相对路径 | 用途 |
| --- | --- | --- |
| Stage 1 epoch 150 | `checkpoints_l1_balanced_5types/restoration_epoch_150.pt` | 初始化 Stage 2 和 Stage 3 |
| Stage 2 epoch 120 完整断点 | `stage2_results_fresh/stage2_ep120_full.pth` | 初始化 Stage 3 |
| Stage 3 epoch 27 | `stage3_results/stage3_ep027.pth` | `test.py` 推理与评估 |

完整的 Stage 3 权重可以单独供 `test.py` 推理，不需要另外传 Stage 1 权重。没有可核实的公开权重下载地址，因此此处不提供下载链接。

## 三阶段训练

```bash
python stage1.py --data-root datasets/DDL

python stage2.py --data-root datasets/DDL \
  --stage1-ckpt checkpoints_l1_balanced_5types/restoration_epoch_150.pt

python stage3.py --data-root datasets/DDL \
  --stage1-ckpt checkpoints_l1_balanced_5types/restoration_epoch_150.pt \
  --stage2-ckpt stage2_results_fresh/stage2_ep120_full.pth
```

Stage 1 默认训练 150 epoch，每 5 轮将权重保存至 `checkpoints_l1_balanced_5types/`，验证图保存至 `val_results_l1_balanced_5types/`。Stage 2 默认训练 150 epoch，权重及完整断点保存至 `stage2_results_fresh/`。Stage 3 默认训练 30 epoch，前 8 轮执行第一阶段冻结策略，之后继续微调；结果保存至 `stage3_results/`。Stage 1/2 如需从完整断点续训，显式传入 `--resume <断点路径>`。三个脚本均可用 `--device cpu` 或 `--device cuda:0` 指定设备。

## 推理和评估

`test.py` 按同名文件配对红外与可见光图，生成融合图并统计指标。单数据集示例：

```bash
python test.py \
  --stage3_ckpt stage3_results/stage3_ep027.pth \
  --dataset_root datasets/example \
  --ir_dir ir --vis_dir vis \
  --dataset_name example \
  --results_root outputs/evaluation
```

这里 `datasets/example` 是用户自行准备的数据目录，包含 `ir/` 和 `vis/` 两个子目录，本仓库没有附带样例图像。融合图写入 `outputs/evaluation/fused_stage3_ep027/example/`；逐图和汇总结果分别写入 `all_details.csv`、`all_summary.csv`，运行配置写入 `run_config.json`。无干净可见光 GT 时，脚本计算 EN、SD、SF、AG，以及可用时的 CPBD、MUSIQ、BRISQUE。

如数据还包含干净可见光图像，可加 `--gt_dir Visible_gt`，脚本会按三路同名文件配对并计算更多指标。多个数据集可改用 `--dataset_config configs/inference_datasets.example.json`；运行前请将示例配置中的路径改成实际位置。`--dataset_root` 和 `--dataset_config` 必须二选一。脚本只保存融合图；如果需要其他输出，请查看模型前向返回的 `restored_image`。

## 已知限制

- 仓库不包含数据集、权重、退化生成代码或可确认适合公开分发的成对图像样例。
- Stage 1 的恢复输出将解码结果与退化 VIS 相加；Stage 2/3 的 `restored_image` 直接取解码结果。此差异来自现有实现。
- 原工作区未发现主项目许可证文件，因此这里未添加许可证。
- 本次整理验证了代码导入、权重加载和单张真实图像的推理；未运行长时间训练或完整数据集复算。
