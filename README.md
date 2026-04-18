# HW1: 三层神经网络分类器 — EuroSAT 地表覆盖分类

从零开始（仅用 NumPy）手工实现三层 MLP，在 EuroSAT RGB 数据集上完成 10 类土地覆盖分类。

---

## 项目结构

```
eurosat-mlp/
├── src/
│   ├── data.py          # 数据加载、归一化、分层划分
│   ├── model.py         # 激活函数、Linear 层、损失、SGD、MLP 模型
│   ├── train.py         # 训练循环 + 训练入口
│   ├── test.py          # 测试评估、可视化 + 测试入口
│   ├── search.py        # 超参数搜索（网格/随机）
│   └── grad_check.py    # 数值梯度验证工具
├── outputs/             # 训练输出图表
├── search_results.csv   # 超参数搜索结果
├── README.md
└── .gitignore
```

**注意**：数据集（`EuroSAT_RGB/`）和模型权重（`best_model.npz`）不包含在 repo 中，请见下方说明。

---

## 网络结构

```
Input (12288 = 64×64×3)
  → Linear(12288 → hidden_dim1) → Activation
  → Linear(hidden_dim1 → hidden_dim2) → Activation
  → Linear(hidden_dim2 → 10)
  → Softmax (仅用于损失计算)
```

本项目完全基于 NumPy 手工实现前向传播与反向传播，不使用 PyTorch / TensorFlow / JAX。

---

## 环境依赖

```
Python >= 3.10
numpy
Pillow
matplotlib
```

```bash
pip install numpy Pillow matplotlib
```

---

## 数据集

[EuroSAT](https://github.com/phelber/EuroSAT)：10 类卫星遥感图像，共 27000 张，每张 64×64 RGB。下载后解压至项目根目录，保持 `EuroSAT_RGB/` 的目录结构。

---

## 训练

```bash
cd src

# 使用默认超参数
python train.py

# 自定义超参数
python train.py \
  --data_dir ../EuroSAT_RGB \
  --hidden_dim1 2048 \
  --hidden_dim2 512 \
  --activation relu \
  --lr 0.05 \
  --decay_rate 0.5 \
  --decay_steps 10 \
  --weight_decay 1e-3 \
  --epochs 100 \
  --batch_size 256 \
  --save_path ../best_model.npz \
  --output_dir ../outputs
```

训练完成后，`outputs/` 下会生成：
- `training_curves.png` — 训练/验证 Loss 及 Accuracy 曲线
- `weights_layer1.png`  — 第一层权重可视化
- `norm_stats.npz`      — 归一化统计量（测试时复用）

---

## 测试

```bash
cd src

python test.py \
  --weight_path ../best_model.npz \
  --hidden_dim1 2048 \
  --hidden_dim2 512 \
  --activation relu \
  --data_dir ../EuroSAT_RGB \
  --output_dir ../outputs
```

输出：
- 测试集分类 Accuracy
- 各类别 Precision / Recall
- 混淆矩阵热图 `confusion_matrix.png`
- 错例分析图 `error_examples.png`

---

## 超参数搜索

```bash
cd src
python search.py
```

结果保存至根目录 `search_results.csv`。

---

## 梯度验证

```bash
cd src
python grad_check.py
```

使用有限差分（ε=1e-5）逐参数检验，确认反向传播实现正确。

---

## 模型权重下载

训练好的权重文件：**[best_model.npz](https://drive.google.com/file/d/1whVHOQhbG8Wre0k1WFUQ2fVbvDsaLSEs/view?usp=sharing)**

下载后放置于项目根目录即可直接运行测试。
