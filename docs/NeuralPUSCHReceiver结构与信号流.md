# NeuralPUSCHReceiver 结构与信号流说明

## 1. 总体结构

`NeuralPUSCHReceiver` 是 5G NR PUSCH 的端到端神经网络接收机，集成了信道估计、神经检测、传输块解码等功能。其核心结构如下：

- **初始信道估计**（LS Channel Estimator）
- **神经接收机核心**（CGNNOFDM → CGNN）
- **传输块解码**（TBDecoder）

---

## 2. 信号流与维度变化

### 输入
- `y`：接收信号 `[batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]` (复数)
- `active_tx`：激活用户掩码 `[batch, num_tx]`
- `b`：原始比特（训练时）`[batch, num_tx, tb_size]`
- `h`：真实信道（训练时）`[batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]`
- `mcs_ue_mask`：MCS掩码 `[batch, max_num_tx, len(mcs_index)]`

### 主要流程

#### 1️⃣ 初始信道估计
- 代码：`estimate_channel()`
- 输入：`y`
- 输出：`h_hat` `[batch, num_tx, num_effective_subcarriers, num_ofdm_symbols, 2*num_rx_ant]`（实部虚部拼接）

#### 2️⃣ 神经接收机核心（CGNNOFDM）
- 代码：`self._neural_rx = CGNNOFDM(...)`
- 主要步骤：
  - **资源网格解映射**（ResourceGridDemapper）：提取数据 RE
  - **CGNN**：
    - 状态初始化（StateInit）
    - 多次迭代（CGNNIt）：聚合用户状态、状态更新
    - 读出 LLR（ReadoutLLRs）和 refined 信道估计（ReadoutChEst）
  - 输出：
    - `llr` `[batch, num_tx, num_data_symbols*num_bits_per_symbol]`
    - `h_hat_refined` `[batch, num_tx, num_effective_subcarriers, num_ofdm_symbols, 2*num_rx_ant]`

#### 3️⃣ 传输块解码（TBDecoder）
- 输入：`llr`
- 输出：
  - `b_hat` `[batch, num_tx, tb_size]`
  - `tb_crc_status` `[batch, num_tx]`

#### 4️⃣ 损失计算（训练时）
- 输出：
  - `loss_data`：BCE 损失
  - `loss_chest`：MSE 损失（信道估计）

---

## 3. 关键代码位置

- `NeuralPUSCHReceiver` 类定义：`utils/neural_rx.py` (约1335行)
- 信道估计：`estimate_channel()`
- 神经网络主干：`CGNNOFDM`、`CGNN`、`StateInit`、`CGNNIt`、`ReadoutLLRs`、`ReadoutChEst`
- 传输块解码：`TBDecoder`

---

## 4. 维度变化示例（以4RB为例，可适配任意RB数）

| 阶段 | 输入维度 | 输出维度 | 说明 |
|------|----------|----------|------|
| 输入 | `[B, Rx, RxAnt, 14, F]` | | 原始接收信号 |
| LS信道估计 | `[B, Rx, RxAnt, 14, F]` | `[B, Tx, F, 14, 2*RxAnt]` | 实部虚部拼接 |
| 预处理 | `[B, 1, RxAnt, 14, F]` | `[B, F, 14, 2*RxAnt]` | 转置、实虚分开 |
| 状态初始化 | `[B, Tx, F, 14, 2*RxAnt]` | `[B, Tx, F, 14, d_s]` | 多层卷积 |
| 迭代 | `[B, Tx, F, 14, d_s]` | `[B, Tx, F, 14, d_s]` | 聚合+更新 |
| 读出LLR | `[B, Tx, F, 14, d_s]` | `[B, Tx, F, 14, bits/sym]` | MLP |
| 解映射 | `[B, Tx, F, 14, bits/sym]` | `[B, Tx, N_bits]` | |
| TB解码 | `[B, Tx, N_bits]` | `[B, Tx, tb_size]` | |

**注：** F=子载波数=RB数×12，所有维度均由配置文件参数自动适配。

---

## 5. 是否支持非4RB？

- ✅ 支持！所有关键层和信号流都能适配不同RB数，只要配置文件设置正确，模型训练时也是该RB数。
- 只要模型是在目标RB数下训练的，推理/评估时无需改代码即可处理非4RB信号。

---

## 6. 主要参数说明

| 参数 | 说明 |
|------|------|
| `num_it` | CGNN 迭代次数（默认5） |
| `d_s` | 状态向量维度（默认32） |
| `num_units_init` | 初始化网络隐藏单元 |
| `num_units_agg` | 聚合网络隐藏单元 |
| `num_units_state` | 状态更新网络隐藏单元 |
| `num_units_readout` | 读出网络隐藏单元 |
| `num_bp_iter` | LDPC BP 迭代次数 |
| `initial_chest` | 初始信道估计方法（如'ls'） |
| `mask_pilots` | 是否屏蔽导频 |

---

## 7. 总结

- NeuralPUSCHReceiver 是端到端的 5G NR 神经接收机，集成信道估计、神经检测、FEC解码。
- 信号流高度参数化，支持任意RB数（只要配置和训练时一致）。
- 结构核心为 LS信道估计 + CGNN图神经网络 + TB解码。
- 适合多用户、多MCS、可变信道等复杂场景。

---

如需进一步分析某一层的实现或某一维度的详细变化，请查阅 `utils/neural_rx.py` 或联系开发者。
