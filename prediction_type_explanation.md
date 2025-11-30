# Prediction Type 详细解释

## 概述

`prediction_type` 是扩散模型调度器（scheduler）中的一个关键参数，它决定了模型在训练和推理时预测的目标是什么。不同的预测类型会影响模型的训练目标和去噪过程。

## 三种预测类型

### 1. `"epsilon"` (噪声预测) - 默认值

**定义**：模型直接预测添加到样本中的噪声 ε。

**数学表示**：
- 在时间步 t，模型预测：`model_output = ε_t`
- 其中 `ε_t` 是添加到干净样本 `x_0` 上的噪声

**去噪公式**（从模型输出计算原始样本）：
```
pred_original_sample = (sample - sqrt(β_t) * model_output) / sqrt(α_t)
```

其中：
- `sample` = `x_t`（当前带噪声的样本）
- `α_t` = `alphas_cumprod[t]`（累积的 alpha 值）
- `β_t` = `1 - α_t`（累积的 beta 值）

**特点**：
- 最常用的预测类型，也是默认值
- 直观：模型学习预测需要移除的噪声
- 在 DDPM 和大多数 Stable Diffusion 模型中使用

**代码示例**（来自 `scheduling_ddim.py`）：
```python
if self.config.prediction_type == "epsilon":
    pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    pred_epsilon = model_output
```

---

### 2. `"sample"` (样本预测)

**定义**：模型直接预测去噪后的干净样本 `x_0`。

**数学表示**：
- 在时间步 t，模型预测：`model_output = x_0`
- 模型直接输出最终的去噪结果

**去噪公式**（从模型输出计算噪声）：
```
pred_original_sample = model_output
pred_epsilon = (sample - sqrt(α_t) * pred_original_sample) / sqrt(β_t)
```

**特点**：
- 模型直接学习预测最终结果
- 在某些架构中可能更稳定
- 较少使用，但在某些特定模型（如 SDXL）中会用到

**代码示例**：
```python
elif self.config.prediction_type == "sample":
    pred_original_sample = model_output
    pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
```

---

### 3. `"v_prediction"` (速度预测)

**定义**：模型预测一个"速度"参数 v，它是噪声和样本的线性组合。

**数学表示**：
- 速度 v 定义为：`v = sqrt(α_t) * ε - sqrt(β_t) * x_t`
- 在时间步 t，模型预测：`model_output = v_t`

**去噪公式**（从速度计算原始样本和噪声）：
```
pred_original_sample = sqrt(α_t) * sample - sqrt(β_t) * model_output
pred_epsilon = sqrt(α_t) * model_output + sqrt(β_t) * sample
```

**速度的计算公式**（用于训练）：
```python
velocity = sqrt(α_t) * noise - sqrt(β_t) * sample
```

**特点**：
- 由 Imagen Video 论文（2022）提出（见论文第 2.4 节）
- 在某些情况下可以提供更稳定的训练
- 在 Stable Diffusion 2.0 的某些版本中使用（特别是训练步数 > 875000 的模型）
- 速度参数在数值上可能比直接预测噪声更稳定

**代码示例**：
```python
elif self.config.prediction_type == "v_prediction":
    pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
```

**速度计算函数**（`get_velocity`）：
```python
def get_velocity(self, sample, noise, timesteps):
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
    velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    return velocity
```

---

## 数学关系总结

三种预测类型可以相互转换：

### 从 epsilon 转换：
- **到 sample**：`x_0 = (x_t - sqrt(β_t) * ε) / sqrt(α_t)`
- **到 v**：`v = sqrt(α_t) * ε - sqrt(β_t) * x_t`

### 从 sample 转换：
- **到 epsilon**：`ε = (x_t - sqrt(α_t) * x_0) / sqrt(β_t)`
- **到 v**：`v = sqrt(α_t) * (x_t - sqrt(α_t) * x_0) / sqrt(β_t) - sqrt(β_t) * x_t`

### 从 v 转换：
- **到 epsilon**：`ε = sqrt(α_t) * v + sqrt(β_t) * x_t`
- **到 sample**：`x_0 = sqrt(α_t) * x_t - sqrt(β_t) * v`

---

## 使用场景

### 何时使用 `"epsilon"`：
- ✅ 大多数 Stable Diffusion 1.x 模型
- ✅ 默认选择，兼容性最好
- ✅ 训练新模型时的常见选择

### 何时使用 `"sample"`：
- ✅ 某些 SDXL 变体
- ✅ 需要直接预测最终结果的场景
- ⚠️ 使用较少，需要确认模型支持

### 何时使用 `"v_prediction"`：
- ✅ Stable Diffusion 2.0 的某些版本（特别是训练步数 > 875000）
- ✅ 基于 Imagen Video 的模型
- ✅ 需要更稳定训练的场景
- ✅ 某些视频生成模型

---

## 训练时的差异

在训练时，不同预测类型的目标值不同：

```python
# epsilon 预测
if prediction_type == "epsilon":
    target = noise  # 直接预测噪声

# v_prediction 预测
elif prediction_type == "v_prediction":
    target = scheduler.get_velocity(latents, noise, timesteps)
    # target = sqrt(α_t) * noise - sqrt(β_t) * latents

# sample 预测
elif prediction_type == "sample":
    target = latents  # 预测去噪后的潜在表示
```

损失权重也可能不同：
- **epsilon**：`loss_weight = 1 / snr`
- **v_prediction**：`loss_weight = 1 / (snr + 1)`

---

## 实际应用建议

1. **检查模型配置**：使用预训练模型时，查看模型的 `scheduler.config.prediction_type`
2. **保持一致性**：训练和推理时使用相同的 `prediction_type`
3. **Stable Diffusion 2.0**：某些版本使用 `v_prediction`，需要特别设置
4. **默认值**：如果不确定，使用 `"epsilon"` 通常是最安全的选择

---

## 参考文献

- **Imagen Video 论文**：https://huggingface.co/papers/2210.02303（v_prediction 的原始来源）
- **DDPM 论文**：https://huggingface.co/papers/2006.11239（epsilon 预测）
- **DDIM 论文**：https://huggingface.co/papers/2010.02502（采样方法）

---

## 代码位置

在 diffusers 库中，这些转换逻辑主要位于：
- `src/diffusers/schedulers/scheduling_ddim.py`（第 448-461 行）
- `src/diffusers/schedulers/scheduling_ddpm.py`（类似位置）
- 各种调度器的 `step()` 方法中
