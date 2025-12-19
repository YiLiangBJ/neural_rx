# NeuralPUSCHReceiver 深度解析（结合代码）

## 1. 架构概览

`NeuralPUSCHReceiver` 是一个完整的 5G NR PUSCH 端到端神经接收机，位于 `utils/neural_rx.py` 第 1335 行。其设计哲学是将传统接收机的各个模块（信道估计、均衡、解调、解码）替换为可训练的神经网络。

### 核心组件

```python
class NeuralPUSCHReceiver(Layer):
    def __init__(self, sys_parameters, training=False, **kwargs):
        # 1. TB 编码器/解码器（每个 MCS 一个）
        self._tb_encoders = []
        self._tb_decoders = []
        
        # 2. LS 信道估计器
        self._ls_est = PUSCHLSChannelEstimator(...)
        
        # 3. 神经接收机核心
        self._neural_rx = CGNNOFDM(...)
```

---

## 2. 初始化详解

### 2.1 多 MCS 支持

```python
self._num_mcss_supported = len(sys_parameters.mcs_index)
for mcs_list_idx in range(self._num_mcss_supported):
    self._tb_encoders.append(
        self._sys_parameters.transmitters[mcs_list_idx]._tb_encoder)
    
    self._tb_decoders.append(
        TBDecoder(self._tb_encoders[mcs_list_idx],
                  num_bp_iter=sys_parameters.num_bp_iter,
                  cn_type=sys_parameters.cn_type))
```

**解释**：
- 系统支持多种 MCS（调制编码方案），如 QPSK、16-QAM、64-QAM
- 每个 MCS 需要独立的编码器/解码器，因为码字长度、调制阶数不同
- `_tb_encoders` 用于训练时重新编码标签
- `_tb_decoders` 用于推理时解码 LLR 为比特

### 2.2 预编码矩阵处理

```python
if hasattr(sys_parameters.transmitters[0], "_precoder"):
    self._precoding_mat = sys_parameters.transmitters[0]._precoder._w
else:
    self._precoding_mat = tf.ones([sys_parameters.max_num_tx,
                                   sys_parameters.num_antenna_ports, 1], 
                                   tf.complex64)
```

**解释**：
- 预编码矩阵用于多天线发射端的空间复用
- 训练时需要用它处理真实信道 `h`，将物理信道转换为等效信道
- 如果没有预编码器，使用单位矩阵（相当于不做预编码）

### 2.3 LS 信道估计器

```python
rg = sys_parameters.transmitters[0]._resource_grid
pc = sys_parameters.pusch_configs[0][0]
self._ls_est = PUSCHLSChannelEstimator(
    resource_grid=rg,
    dmrs_length=pc.dmrs.length,
    dmrs_additional_position=pc.dmrs.additional_position,
    num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
    interpolation_type="nn")  # 最近邻插值
```

**解释**：
- LS (Least Squares) 估计器基于 DMRS 导频进行信道估计
- `interpolation_type="nn"` 使用最近邻插值，将导频位置的估计扩展到所有子载波
- 这是神经网络的"初始化"，提供粗糙但结构化的信道信息

### 2.4 Layer Demapper

```python
self._layer_demappers = []
for mcs_list_idx in range(self._num_mcss_supported):
    self._layer_demappers.append(
        LayerDemapper(
            self._sys_parameters.transmitters[mcs_list_idx]._layer_mapper,
            sys_parameters.transmitters[mcs_list_idx]._num_bits_per_symbol))
```

**解释**：
- LayerDemapper 是 MIMO 层解映射器
- 在多层传输时，将不同空间流的符号分离
- 每个 MCS 有不同的 `num_bits_per_symbol`（如 QPSK=2, 16-QAM=4, 64-QAM=6）

### 2.5 神经接收机核心

```python
self._neural_rx = CGNNOFDM(
    sys_parameters,
    max_num_tx=sys_parameters.max_num_tx,
    training=training,
    num_it=sys_parameters.num_nrx_iter,        # 迭代次数，如 5
    d_s=sys_parameters.d_s,                    # 状态维度，如 32
    num_units_init=sys_parameters.num_units_init,      # [64]
    num_units_agg=sys_parameters.num_units_agg,        # [[64]]
    num_units_state=sys_parameters.num_units_state,    # [[64]]
    num_units_readout=sys_parameters.num_units_readout, # [64]
    layer_demappers=self._layer_demappers,
    layer_type_dense=sys_parameters.layer_type_dense,
    layer_type_conv=sys_parameters.layer_type_conv,
    layer_type_readout=sys_parameters.layer_type_readout,
    dtype=sys_parameters.nrx_dtype)
```

**解释**：
- `CGNNOFDM` 是神经接收机的主体，包含 CGNN（图神经网络）
- `num_it` 控制迭代次数，每次迭代精炼一次检测结果
- `d_s` 是每个资源元素的状态向量维度
- `num_units_*` 控制各子网络的隐藏层大小

---

## 3. 信道估计方法详解

### 3.1 estimate_channel 函数

```python
def estimate_channel(self, y, num_tx):
    # y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_subcarriers]
    
    if self._sys_parameters.initial_chest == 'ls':
        if self._sys_parameters.mask_pilots:
            raise ValueError("Cannot use initial channel estimator if " \
                            "pilots are masked.")
        
        # LS 估计
        h_hat, _ = self._ls_est([y, 1e-1])
        # 输出: [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
        #        num_ofdm_symbols, num_effective_subcarriers]
        
        # 维度重塑
        h_hat = h_hat[:,0,:,:num_tx,0]
        # → [batch_size, num_rx_ant, num_tx, num_ofdm_symbols, num_subcarriers]
        
        h_hat = tf.transpose(h_hat, [0, 2, 4, 3, 1])
        # → [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        
        # 复数 → 实数
        h_hat = tf.concat([tf.math.real(h_hat), tf.math.imag(h_hat)], axis=-1)
        # → [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
    
    elif self._sys_parameters.initial_chest == None:
        h_hat = None  # 无导频通信
    
    return h_hat
```

**关键点**：
1. **LS 估计原理**：在导频位置，使用已知导频符号计算 `h = y / pilot`
2. **最近邻插值**：将导频位置的估计复制到最近的数据子载波
3. **复数到实数**：神经网络处理实数，将复数信道分为实部和虚部
4. **Pilotless 模式**：`initial_chest=None` 时不使用初始估计，完全由神经网络学习

### 3.2 预处理真实信道（训练用）

```python
def preprocess_channel_ground_truth(self, h):
    # h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,
    #     num_ofdm_symbols, num_effective_subcarriers]
    
    h = tf.squeeze(h, axis=1)  # 移除 num_rx 维度（假设=1）
    
    # 转置
    h = tf.transpose(h, perm=[0,2,5,4,1,3])
    # → [batch, num_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant, num_tx_ant]
    
    # 应用预编码矩阵
    w = insert_dims(tf.expand_dims(self._precoding_mat, axis=0), 2, 2)
    h = tf.squeeze(tf.matmul(h, w), axis=-1)
    # → [batch, num_tx, num_ofdm_symbols, num_subcarriers, num_rx_ant]
    
    # 复数到实数
    h = tf.concat([tf.math.real(h), tf.math.imag(h)], axis=-1)
    # → [batch, num_tx, num_ofdm_symbols, num_subcarriers, 2*num_rx_ant]
    
    return h
```

**解释**：
- 训练时需要真实信道计算损失
- 预编码矩阵将物理 MIMO 信道转换为等效单流信道
- 最后的形状与 LS 估计的 `h_hat` 保持一致

---

## 4. 主流程 call 函数详解

### 4.1 训练模式

```python
if self._training:
    y, active_tx, b, h, mcs_ue_mask = inputs
    
    # 重新编码原始比特（生成训练标签）
    if len(mcs_arr_eval)==1 and not isinstance(b, list):
        b = [b]
    bits = []
    for idx in range(len(mcs_arr_eval)):
        bits.append(
            self._sys_parameters.transmitters[mcs_arr_eval[idx]]._tb_encoder(b[idx]))
    
    # 初始信道估计
    num_tx = tf.shape(active_tx)[1]
    h_hat = self.estimate_channel(y, num_tx)
    
    # 处理真实信道
    if h is not None:
        h = self.preprocess_channel_ground_truth(h)
    
    # 调用神经接收机
    losses = self._neural_rx((y, h_hat, active_tx, bits, h, mcs_ue_mask),
                             mcs_arr_eval)
    return losses  # (loss_data, loss_chest)
```

**训练流程**：
1. 输入原始信息比特 `b`（未编码的）
2. 用 TB Encoder 重新编码为 `bits`（编码后的，含 CRC、LDPC 等）
3. LS 估计得到 `h_hat`
4. 处理真实信道 `h`（用于计算损失）
5. 神经网络输出两个损失：
   - `loss_data`：LLR 与编码比特的 BCE 损失
   - `loss_chest`：信道估计与真实信道的 MSE 损失

### 4.2 推理模式

```python
else:
    y, active_tx = inputs
    
    # 初始信道估计
    num_tx = tf.shape(active_tx)[1]
    h_hat = self.estimate_channel(y, num_tx)
    
    # 调用神经接收机
    llr, h_hat_refined = self._neural_rx(
        (y, h_hat, active_tx),
        [mcs_arr_eval[0]],
        mcs_ue_mask_eval=mcs_ue_mask_eval)
    
    # TB 解码
    b_hat, tb_crc_status = self._tb_decoders[mcs_arr_eval[0]](llr)
    
    return b_hat, h_hat_refined, h_hat, tb_crc_status
```

**推理流程**：
1. LS 估计得到 `h_hat`
2. 神经网络输出 `llr` 和精炼的 `h_hat_refined`
3. TB Decoder 解码 LLR 得到信息比特 `b_hat`
4. 返回解码比特、精炼信道估计、初始信道估计、CRC 状态

---

## 5. CGNN 核心网络详解

### 5.1 整体结构

```python
class CGNN(Model):
    def __init__(self, num_bits_per_symbol, num_rx_ant, num_it, d_s, ...):
        # 状态初始化网络
        self._s_init = []
        for _ in num_bits_per_symbol:
            self._s_init.append(StateInit(d_s, num_units_init, ...))
        
        # 迭代块（每次迭代独立的网络）
        self._iterations = []
        for i in range(num_it):
            it = CGNNIt(d_s, num_units_agg[i], num_units_state[i], ...)
            self._iterations.append(it)
        
        # 读出网络
        self._readout_llrs = []
        for num_bits in num_bits_per_symbol:
            self._readout_llrs.append(ReadoutLLRs(num_bits, ...))
        
        self._readout_chest = ReadoutChEst(num_rx_ant, ...)
```

**设计思想**：
- **迭代检测**：类似 Turbo 迭代，每次迭代精炼一次估计
- **无权重共享**：每次迭代使用独立的网络（性能更好）
- **多 MCS 支持**：每个 MCS 有独立的读出层

### 5.2 归一化

```python
def call(self, inputs):
    y, pe, h_hat, active_tx, mcs_ue_mask = inputs
    
    # 归一化（每个 batch 样本单位功率）
    norm_scaling = tf.reduce_mean(tf.square(y), axis=(1,2,3), keepdims=True)
    norm_scaling = tf.math.divide_no_nan(1., tf.sqrt(norm_scaling))
    y = y * norm_scaling
    
    # 同样归一化信道估计
    norm_scaling = tf.expand_dims(norm_scaling, axis=1)
    if h_hat is not None:
        h_hat = h_hat * norm_scaling
```

**解释**：
- 归一化确保不同 SNR 下输入尺度一致
- 有利于网络训练稳定性和泛化能力
- 信道估计也需要同步归一化

### 5.3 状态初始化

```python
# StateInit 网络结构
class StateInit(Layer):
    def __init__(self, d_s, num_units, layer_type="sepconv", ...):
        self._hidden_conv = []
        for n in num_units:  # 如 [64]
            conv = SeparableConv2D(n, (3,3), padding='same', 
                                   activation='relu', ...)
            self._hidden_conv.append(conv)
        
        self._output_conv = SeparableConv2D(d_s, (3,3), activation=None, 
                                            padding='same', ...)
    
    def call(self, inputs):
        y, pe, h_hat = inputs
        # 拼接输入
        if h_hat is not None:
            z = tf.concat([y, pe, h_hat], axis=-1)
        else:
            z = tf.concat([y, pe], axis=-1)
        
        # 卷积处理
        for conv in self._hidden_conv:
            z = conv(z)
        s0 = self._output_conv(z)
        
        return s0  # [batch, num_tx, subcarriers, symbols, d_s]
```

**解释**：
- 输入：接收信号 `y`、位置编码 `pe`、初始信道估计 `h_hat`
- 使用 **SeparableConv2D**（可分离卷积）：
  - 先做空间卷积（3x3），再做逐点卷积
  - 参数量更少，计算更快
- 输出：每个资源元素的初始状态向量 `s0`（维度 `d_s`）

### 5.4 迭代更新

```python
# CGNNIt 迭代块
class CGNNIt(Layer):
    def __init__(self, d_s, num_units_agg, num_units_state, ...):
        # 聚合网络
        self._agg = AggregateUserStates(d_s, num_units_agg, ...)
        # 状态更新网络
        self._update = UpdateState(d_s, num_units_state, ...)
    
    def call(self, inputs):
        s, pe, active_tx = inputs
        
        # 聚合其他用户的状态
        a = self._agg([s, active_tx])
        
        # 更新状态
        s = self._update([s, a, pe])
        
        return s
```

**聚合网络详解**：

```python
class AggregateUserStates(Layer):
    def call(self, inputs):
        s, active_tx = inputs
        
        # MLP 处理每个状态
        sp = s
        for layer in self._hidden_layers:  # Dense layers
            sp = layer(sp)
        sp = self._output_layer(sp)
        
        # 屏蔽非激活用户
        active_tx = expand_to_rank(active_tx, tf.rank(sp), axis=-1)
        sp = tf.multiply(sp, active_tx)
        
        # 聚合（移除自身）
        a = tf.reduce_sum(sp, axis=1, keepdims=True) - sp
        
        # 按激活用户数归一化
        p = tf.reduce_sum(active_tx, axis=1, keepdims=True) - 1.
        p = tf.nn.relu(p)
        p = tf.where(p==0., 1., tf.math.divide_no_nan(1., p))
        a = tf.multiply(a, p)
        
        return a  # [batch, num_tx, subcarriers, symbols, d_s]
```

**关键思想**：
- **图神经网络**：每个用户是图中的一个节点
- **消息传递**：聚合其他用户的状态作为"消息"
- **干扰感知**：通过聚合学习多用户干扰模式
- **屏蔽机制**：非激活用户不参与聚合

**状态更新网络**：

```python
class UpdateState(Layer):
    def call(self, inputs):
        s, a, pe = inputs
        
        # 拼接当前状态、聚合状态、位置编码
        z = tf.concat([s, a, pe], axis=-1)
        
        # MLP 更新
        for layer in self._hidden_layers:
            z = layer(z)
        
        # 残差连接
        s_new = self._output_layer(z) + s
        
        return s_new
```

**解释**：
- 输入：当前状态 `s`、聚合状态 `a`、位置编码 `pe`
- 使用 MLP（Dense layers）处理
- **残差连接**：`s_new = f(s, a, pe) + s`，有利于梯度传播

### 5.5 读出网络

```python
class ReadoutLLRs(Layer):
    def call(self, s):
        # s: [batch, num_tx, subcarriers, symbols, d_s]
        
        z = s
        for layer in self._hidden_layers:  # Dense layers
            z = layer(z)
        
        llrs = self._output_layer(z)
        # → [batch, num_tx, subcarriers, symbols, num_bits_per_symbol]
        
        return llrs
```

```python
class ReadoutChEst(Layer):
    def call(self, s):
        z = s
        for layer in self._hidden_layers:
            z = layer(z)
        
        h_hat = self._output_layer(z)
        # → [batch, num_tx, subcarriers, symbols, 2*num_rx_ant]
        
        return h_hat
```

**解释**：
- 从状态向量 `s` 读出 LLR 和信道估计
- LLR 用于后续解码
- 信道估计用于训练时的双损失（double readout）

### 5.6 完整迭代流程

```python
def call(self, inputs):
    y, pe, h_hat, active_tx, mcs_ue_mask = inputs
    
    # 1. 归一化
    # ...
    
    # 2. 状态初始化
    if self._var_mcs_masking:
        s = self._s_init[0]((y, pe, h_hat))
    else:
        # 每个 MCS 独立初始化，按 mcs_ue_mask 加权
        s = self._s_init[0]((y, pe, h_hat)) * expand_to_rank(
            tf.gather(mcs_ue_mask, indices=0, axis=2), 5, axis=-1)
        for idx in range(1, self._num_mcss_supported):
            s = s + self._s_init[idx]((y, pe, h_hat)) * expand_to_rank(
                tf.gather(mcs_ue_mask, indices=idx, axis=2), 5, axis=-1)
    
    # 3. 迭代更新
    llrs = []
    h_hats = []
    for i in range(self._num_it):
        it = self._iterations[i]
        
        # 状态更新
        s = it([s, pe, active_tx])
        
        # 读出（训练时每次迭代都读，推理时只读最后一次）
        if (self._training and self._apply_multiloss) or i==self._num_it-1:
            llrs_ = []
            for idx in range(self._num_mcss_supported):
                if self._var_mcs_masking:
                    llrs__ = self._readout_llrs[0](s)
                    llrs__ = tf.gather(llrs__, 
                                      indices=tf.range(self._num_bits_per_symbol[idx]),
                                      axis=-1)
                else:
                    llrs__ = self._readout_llrs[idx](s)
                llrs_.append(llrs__)
            llrs.append(llrs_)
            h_hats.append(self._readout_chest(s))
    
    return llrs, h_hats
```

---

## 6. CGNNOFDM 包装层

### 6.1 位置编码预计算

```python
class CGNNOFDM(Model):
    def __init__(self, sys_parameters, ...):
        # 预计算位置编码
        rg_type = self._rg.build_type_grid()[:,0]
        pilot_ind = tf.where(rg_type==1)
        
        # 计算每个 RE 到最近导频的距离（时域+频域）
        nearest_pilot_dist_time = ...  # [num_tx, symbols, subcarriers]
        nearest_pilot_dist_freq = ...
        
        # 归一化（零均值，单位方差）
        nearest_pilot_dist_time -= np.mean(...)
        nearest_pilot_dist_time /= np.std(...)
        
        # 拼接
        nearest_pilot_dist = np.stack([nearest_pilot_dist_time, 
                                      nearest_pilot_dist_freq], axis=-1)
        
        self._nearest_pilot_dist = tf.transpose(nearest_pilot_dist, [0,2,1,3])
        # → [num_tx, subcarriers, symbols, 2]
```

**解释**：
- **位置编码**：告诉网络每个 RE 离导频有多远
- 类似 Transformer 的位置编码，但是 2D 的（时域+频域）
- 预计算可以避免每次前向都重新计算

### 6.2 导频屏蔽（Pilotless）

```python
def call(self, inputs, mcs_arr_eval, mcs_ue_mask_eval=None):
    # ...
    
    if self._sys_parameters.mask_pilots:
        rg_type = self._rg.build_type_grid()
        rg_type = tf.expand_dims(rg_type, axis=0)
        rg_type = tf.broadcast_to(rg_type, tf.shape(y))
        
        # 将导频位置置零
        y = tf.where(rg_type==1, tf.constant(0., y.dtype), y)
```

**解释**：
- 无导频通信：训练时屏蔽导频，强迫网络学习盲检测
- 测试时也不使用导频，完全依赖神经网络
- 可以节省导频开销，提高频谱效率

### 6.3 资源网格处理

```python
# 重塑为神经网络输入格式
y = y[:,0]  # 移除 num_rx 维度
y = tf.transpose(y, [0, 3, 2, 1])
# → [batch, subcarriers, symbols, num_rx_ant]

# 复数到实数
y = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1)
# → [batch, subcarriers, symbols, 2*num_rx_ant]

# 位置编码
pe = self._nearest_pilot_dist[:num_tx]
# → [num_tx, subcarriers, symbols, 2]
```

### 6.4 损失计算

```python
if self._training:
    loss_data = tf.constant(0.0, dtype=tf.float32)
    for llrs_ in llrs:  # 遍历每次迭代
        for idx in range(len(indices)):  # 遍历每个 MCS
            # 计算 LLR 与编码比特的 BCE 损失
            # ...
    
    loss_chest = tf.constant(0.0, dtype=tf.float32)
    if h_hats is not None:
        for h_hat_ in h_hats:
            # 计算信道估计与真实信道的 MSE 损失
            # ...
    
    # 只统计激活用户
    active_tx_chest = expand_to_rank(active_tx, tf.rank(loss_chest), axis=-1)
    loss_chest = tf.multiply(loss_chest, active_tx_chest)
    loss_chest = tf.reduce_mean(loss_chest)
    
    return loss_data, loss_chest
```

**解释**：
- **双损失**：既优化 LLR 准确性，也优化信道估计准确性
- **多迭代损失**：可以对每次迭代都计算损失（`apply_multiloss=True`）
- **屏蔽机制**：非激活用户不参与损失计算

---

## 7. 关键设计思想

### 7.1 端到端训练

```
输入比特 b → TB Encoder → bits (标签)
                                ↓
接收信号 y → NeuralRx → LLR ─→ BCE Loss(LLR, bits)
                         ↓
真实信道 h ──────────→ h_hat → MSE Loss(h_hat, h)
```

**优势**：
- 直接优化端到端性能（误比特率）
- 无需手动设计特征提取器
- 可以学习传统方法难以建模的非线性

### 7.2 迭代检测

```
初始化 → 迭代1 → 迭代2 → ... → 迭代5
   s0      s1      s2            s5
           ↓       ↓             ↓
         LLR1    LLR2          LLR5 (最终输出)
```

**类比**：
- 类似 Turbo 迭代解码
- 类似 LDPC BP 迭代
- 每次迭代精炼一次估计

### 7.3 图神经网络

```
用户1 ←─聚合─→ 用户2
  ↓              ↓
状态更新      状态更新
  ↓              ↓
用户1 ←─聚合─→ 用户2
```

**多用户干扰学习**：
- 传统方法：干扰消除、干扰对齐
- 神经方法：学习用户间的交互模式
- 自适应不同用户数、信道条件

### 7.4 位置编码

```
RE(i,j) 的特征 = [y, h_hat, dist_to_pilot_time, dist_to_pilot_freq]
```

**作用**：
- 告诉网络当前 RE 的"绝对位置"
- 信道在不同位置有不同的可靠性（导频附近更准）
- 类似 Transformer 的位置编码

---

## 8. 参数量与复杂度

### 典型配置（nrx_rt）

```python
num_it = 5
d_s = 32
num_units_init = [64]
num_units_agg = [[64]] * 5
num_units_state = [[64]] * 5
num_units_readout = [64]
```

**参数量分布**（约 560K）：
- StateInit: ~10K
- CGNNIt × 5: ~400K
  - AggregateUserStates: ~100K
  - UpdateState: ~100K
  - 每次迭代 ~80K
- ReadoutLLRs: ~40K
- ReadoutChEst: ~10K

### 计算复杂度

- **前向传播**：O(num_it × num_tx × num_subcarriers × num_symbols × d_s²)
- **主要瓶颈**：卷积层（StateInit）和 Dense 层（迭代块）
- **XLA 优化**：通过 JIT 编译加速

---

## 9. 与传统接收机对比

| 模块 | 传统接收机 | 神经接收机 |
|------|-----------|-----------|
| 信道估计 | LS/LMMSE | LS初始化 + 神经网络精炼 |
| 均衡 | MMSE/ZF | 隐式在状态更新中 |
| 解调 | 硬判决/软判决 | 读出 LLR |
| 解码 | LDPC BP | LDPC BP（共用） |
| 多用户检测 | SIC/K-Best | 图聚合 |
| 优化目标 | 手工设计 | 端到端训练（BER） |

---

## 10. 适用场景与限制

### 适用场景

✅ 多用户 MIMO（干扰复杂）  
✅ 低 SNR（传统方法性能差）  
✅ 特定信道（可以针对性训练）  
✅ 可变 MCS  
✅ 实时推理（XLA 优化后）

### 限制

❌ 需要大量训练数据  
❌ 泛化能力取决于训练覆盖  
❌ 对新信道类型可能需要重新训练  
❌ 模型大小（1-2MB）比传统算法大  
❌ 推理延迟（迭代 5 次）比单次传统算法慢

---

## 11. 总结

`NeuralPUSCHReceiver` 是一个精心设计的端到端神经接收机：

1. **模块化设计**：LS 估计 + CGNN + TB 解码
2. **迭代检测**：5 次迭代精炼
3. **图神经网络**：多用户干扰学习
4. **双损失训练**：LLR + 信道估计
5. **灵活配置**：支持多 MCS、可变用户数、pilotless
6. **工程优化**：归一化、残差连接、位置编码、XLA 编译

通过神经网络学习复杂的信号处理任务，在某些场景下超越传统方法的性能上限。

---

**代码位置**：`utils/neural_rx.py`  
**核心类**：`NeuralPUSCHReceiver`、`CGNNOFDM`、`CGNN`、`StateInit`、`CGNNIt`
