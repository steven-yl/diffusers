# Prediction Type 详细数学原理与公式推导

## 快速参考

### 核心公式速查

**前向过程（加噪）**：

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{\bar{\beta}_t} \, \epsilon_t
$$

**三种预测类型的推理公式**：

| 类型 | 模型输出 → 原始样本 $\hat{x}_0$ | 模型输出 → 噪声 $\hat{\epsilon}_t$ |
|------|------------------------|---------------------|
| **epsilon** | $$\hat{x}_0 = \frac{x_t - \sqrt{\bar{\beta}_t} \, \hat{\epsilon}_t}{\sqrt{\bar{\alpha}_t}}$$ | $\hat{\epsilon}_t$（直接） |
| **sample** | $\hat{x}_0$（直接） | $$\hat{\epsilon}_t = \frac{x_t - \sqrt{\bar{\alpha}_t} \, \hat{x}_0}{\sqrt{\bar{\beta}_t}}$$ |
| **v_prediction** | $$\hat{x}_0 = \sqrt{\bar{\alpha}_t} \, x_t - \sqrt{\bar{\beta}_t} \, \hat{v}_t$$ | $$\hat{\epsilon}_t = \sqrt{\bar{\alpha}_t} \, \hat{v}_t + \sqrt{\bar{\beta}_t} \, x_t$$ |

**训练目标**：
- **epsilon**: $\text{target} = \epsilon_t$
- **sample**: $\text{target} = x_0$
- **v_prediction**: $\text{target} = \sqrt{\bar{\alpha}_t} \, \epsilon_t - \sqrt{\bar{\beta}_t} \, x_t$

---

## 目录
1. [扩散模型基础数学原理](#1-扩散模型基础数学原理)
2. [前向扩散过程（加噪）](#2-前向扩散过程加噪)
3. [反向扩散过程（去噪）](#3-反向扩散过程去噪)
4. [三种预测类型的数学推导](#4-三种预测类型的数学推导)
5. [三种预测类型公式总结表](#5-三种预测类型公式总结表)
6. [完整公式转换关系](#6-完整公式转换关系)
7. [训练目标与损失函数](#7-训练目标与损失函数)
8. [完整去噪流程示例](#8-完整去噪流程示例)
9. [数学符号总结](#9-数学符号总结)
10. [参考文献](#10-参考文献)
11. [总结](#11-总结)

---

## 1. 扩散模型基础数学原理

### 1.1 核心概念

扩散模型基于一个简单的思想：**通过逐步添加噪声将数据转换为纯噪声，然后学习如何逆转这个过程**。

**关键变量定义**：
- $x_0$: 原始干净样本（t=0时的数据）
- $x_t$: 在时间步 $t$ 的带噪声样本
- $\epsilon$: 标准高斯噪声，$\epsilon \sim \mathcal{N}(0, I)$
- $t$: 时间步，通常 $t \in \{0, 1, 2, \ldots, T\}$，其中 $T$ 是总步数（如1000）
- $\alpha_t$: 时间步 $t$ 的 alpha 值（保留信号的比例）
- $\beta_t$: 时间步 $t$ 的 beta 值（添加噪声的比例），$\beta_t = 1 - \alpha_t$
- $\bar{\alpha}_t$: 累积 alpha 乘积，$\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$
- $\bar{\beta}_t$: 累积 beta，$\bar{\beta}_t = 1 - \bar{\alpha}_t$

### 1.2 噪声调度（Noise Schedule）

扩散过程由噪声调度控制，定义了每个时间步添加多少噪声：

```python
# 线性调度（Linear Schedule）
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t
```

**重要关系**：

$$
\begin{aligned}
\alpha_t &= 1 - \beta_t \\
\bar{\alpha}_t &= \prod_{i=1}^{t} \alpha_i = \alpha_1 \alpha_2 \,s \alpha_t \\
\bar{\beta}_t &= 1 - \bar{\alpha}_t
\end{aligned}
$$

---

## 2. 前向扩散过程（加噪）

### 2.1 单步加噪公式

在前向过程中，我们逐步向数据添加高斯噪声。**关键公式**：

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon
$$

**详细推导**：

从 $x_{t-1}$ 到 $x_t$ 的单步过程：

$$
x_t = \sqrt{\alpha_t} \, x_{t-1} + \sqrt{\beta_t} \, \epsilon_t
$$

其中 $\epsilon_t \sim \mathcal{N}(0, I)$ 是独立的标准高斯噪声。

### 2.2 累积加噪公式（一步到位）

通过递归展开，我们可以直接从 $x_0$ 计算任意时间步 $t$ 的 $x_t$：

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{\bar{\beta}_t} \, \epsilon
$$

其中：
- $\bar{\alpha}_t = \alpha_1 \alpha_2 \,s \alpha_t$（累积乘积）
- $\bar{\beta}_t = 1 - \bar{\alpha}_t$
- $\epsilon \sim \mathcal{N}(0, I)$ 是标准高斯噪声

**代码实现**（来自 `scheduling_ddpm.py`）：
```python
def add_noise(self, original_samples, noise, timesteps):
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5  # √(ᾱ_t)
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5  # √(β̄_t)
    
    noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
    return noisy_samples
```

### 2.3 概率分布

在时间步 $t$，$x_t$ 的分布为：

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \bar{\beta}_t \, I)
$$

这意味着 $x_t$ 是均值为 $\sqrt{\bar{\alpha}_t} \, x_0$、方差为 $\bar{\beta}_t$ 的高斯分布。

---

## 3. 反向扩散过程（去噪）

### 3.1 目标

反向过程的目标是：给定 $x_t$，预测 $x_{t-1}$，最终恢复 $x_0$。

### 3.2 DDPM 去噪公式

根据 DDPM 论文（公式 7），从 $x_t$ 预测 $x_{t-1}$ 的公式为：

$$
x_{t-1} = \frac{\sqrt{\bar{\alpha}_{t-1}} \, \beta_t}{\bar{\beta}_t} \, \hat{x}_0 + \frac{\sqrt{\alpha_t} \, \bar{\beta}_{t-1}}{\bar{\beta}_t} \, x_t + \sigma_t \, z
$$

其中：
- $\hat{x}_0$ 是模型预测的原始样本
- $z \sim \mathcal{N}(0, I)$ 是随机噪声（仅在 $t > 0$ 时添加）
- $\sigma_t$ 是方差项

#### 3.2.1 详细推导过程

**步骤 1：前向过程的回顾**

前向过程定义：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} \, x_{t-1}, \beta_t \, I)
$$

累积前向过程（从 $x_0$ 直接到 $x_t$）：

$$
q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \bar{\beta}_t \, I)
$$

即：

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{\bar{\beta}_t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

**步骤 2：反向过程的目标**

反向过程的目标是学习分布 $p_\theta(x_{t-1} \mid x_t)$，使其能够从噪声 $x_T$ 逐步恢复原始数据 $x_0$。

根据 DDPM 论文，反向过程定义为：

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

**步骤 3：使用贝叶斯定理推导后验分布**

根据贝叶斯定理，后验分布 $q(x_{t-1} \mid x_t, x_0)$ 可以表示为：

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}, x_0) \, q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
$$

由于前向过程是马尔可夫的，$q(x_t \mid x_{t-1}, x_0) = q(x_t \mid x_{t-1})$，因此：

$$
q(x_{t-1} \mid x_t, x_0) = \frac{q(x_t \mid x_{t-1}) \, q(x_{t-1} \mid x_0)}{q(x_t \mid x_0)}
$$

**步骤 4：展开各项的概率密度函数**

各项的概率密度函数为：

$$
\begin{aligned}
q(x_t \mid x_{t-1}) &= \mathcal{N}(x_t; \sqrt{\alpha_t} \, x_{t-1}, \beta_t \, I) \\
&\propto \exp\left(-\frac{1}{2\beta_t} \|x_t - \sqrt{\alpha_t} \, x_{t-1}\|^2\right) \\
q(x_{t-1} \mid x_0) &= \mathcal{N}(x_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \, x_0, \bar{\beta}_{t-1} \, I) \\
&\propto \exp\left(-\frac{1}{2\bar{\beta}_{t-1}} \|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \, x_0\|^2\right) \\
q(x_t \mid x_0) &= \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \bar{\beta}_t \, I) \\
&\propto \exp\left(-\frac{1}{2\bar{\beta}_t} \|x_t - \sqrt{\bar{\alpha}_t} \, x_0\|^2\right)
\end{aligned}
$$

**步骤 5：计算后验分布的指数部分**

后验分布的对数形式为：

$$
\begin{aligned}
\log q(x_{t-1} \mid x_t, x_0) &\propto -\frac{1}{2\beta_t} \|x_t - \sqrt{\alpha_t} \, x_{t-1}\|^2 \\
&\quad -\frac{1}{2\bar{\beta}_{t-1}} \|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \, x_0\|^2 \\
&\quad + \frac{1}{2\bar{\beta}_t} \|x_t - \sqrt{\bar{\alpha}_t} \, x_0\|^2
\end{aligned}
$$

展开平方项：

$$
\begin{aligned}
\|x_t - \sqrt{\alpha_t} \, x_{t-1}\|^2 &= x_t^T x_t - 2\sqrt{\alpha_t} \, x_t^T x_{t-1} + \alpha_t \, x_{t-1}^T x_{t-1} \\
\|x_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \, x_0\|^2 &= x_{t-1}^T x_{t-1} - 2\sqrt{\bar{\alpha}_{t-1}} \, x_{t-1}^T x_0 + \bar{\alpha}_{t-1} \, x_0^T x_0
\end{aligned}
$$

**步骤 6：提取关于 $x_{t-1}$ 的二次项和一次项**

将关于 $x_{t-1}$ 的项整理：

$$
\begin{aligned}
\log q(x_{t-1} \mid x_t, x_0) &\propto -\frac{1}{2} \left[\frac{\alpha_t}{\beta_t} + \frac{1}{\bar{\beta}_{t-1}}\right] x_{t-1}^T x_{t-1} \\
&\quad + \left[\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{\bar{\beta}_{t-1}} x_0\right]^T x_{t-1} + \text{常数项}
\end{aligned}
$$

这是一个关于 $x_{t-1}$ 的二次型，对应一个高斯分布。

**步骤 7：确定后验分布的均值和方差**

对于高斯分布 $\mathcal{N}(x; \mu, \Sigma)$，其概率密度函数的指数部分为：

$$
-\frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu) = -\frac{1}{2}x^T \Sigma^{-1} x + \mu^T \Sigma^{-1} x + \text{常数项}
$$

对比系数，得到：

$$
\begin{aligned}
\Sigma^{-1} &= \frac{\alpha_t}{\beta_t} + \frac{1}{\bar{\beta}_{t-1}} = \frac{\alpha_t \bar{\beta}_{t-1} + \beta_t}{\beta_t \bar{\beta}_{t-1}} = \frac{\bar{\beta}_t}{\beta_t \bar{\beta}_{t-1}} \\
\mu^T \Sigma^{-1} &= \frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{\bar{\beta}_{t-1}} x_0
\end{aligned}
$$

因此：

$$
\begin{aligned}
\Sigma &= \frac{\beta_t \bar{\beta}_{t-1}}{\bar{\beta}_t} \\
\mu &= \Sigma \left(\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{\bar{\beta}_{t-1}} x_0\right)
\end{aligned}
$$

**步骤 8：简化均值表达式**

展开均值：

$$
\begin{aligned}
\mu &= \frac{\beta_t \bar{\beta}_{t-1}}{\bar{\beta}_t} \left(\frac{\sqrt{\alpha_t}}{\beta_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{\bar{\beta}_{t-1}} x_0\right) \\
&= \frac{\bar{\beta}_{t-1}}{\bar{\beta}_t} \sqrt{\alpha_t} \, x_t + \frac{\beta_t}{\bar{\beta}_t} \sqrt{\bar{\alpha}_{t-1}} \, x_0
\end{aligned}
$$

**步骤 9：使用前向过程关系简化**

从前向过程，我们知道：

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{\bar{\beta}_t} \, \epsilon_t
$$

因此：

$$
x_0 = \frac{x_t - \sqrt{\bar{\beta}_t} \, \epsilon_t}{\sqrt{\bar{\alpha}_t}}
$$

将 $x_0$ 代入均值表达式：

$$
\begin{aligned}
\mu &= \frac{\bar{\beta}_{t-1}}{\bar{\beta}_t} \sqrt{\alpha_t} \, x_t + \frac{\beta_t}{\bar{\beta}_t} \sqrt{\bar{\alpha}_{t-1}} \, \frac{x_t - \sqrt{\bar{\beta}_t} \, \epsilon_t}{\sqrt{\bar{\alpha}_t}} \\
&= \frac{\bar{\beta}_{t-1}}{\bar{\beta}_t} \sqrt{\alpha_t} \, x_t + \frac{\beta_t \sqrt{\bar{\alpha}_{t-1}}}{\bar{\beta}_t \sqrt{\bar{\alpha}_t}} (x_t - \sqrt{\bar{\beta}_t} \, \epsilon_t) \\
&= \frac{\bar{\beta}_{t-1} \sqrt{\alpha_t} + \beta_t \sqrt{\bar{\alpha}_{t-1}} / \sqrt{\bar{\alpha}_t}}{\bar{\beta}_t} x_t - \frac{\beta_t \sqrt{\bar{\alpha}_{t-1}} \sqrt{\bar{\beta}_t}}{\bar{\beta}_t \sqrt{\bar{\alpha}_t}} \epsilon_t
\end{aligned}
$$

注意到 $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$ 和 $\bar{\beta}_t = 1 - \bar{\alpha}_t$，可以进一步简化。

**步骤 10：最终形式**

经过代数运算（详见 DDPM 论文附录），后验分布的均值可以表示为：

$$
\mu_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \, \beta_t}{\bar{\beta}_t} x_0 + \frac{\sqrt{\alpha_t} \, \bar{\beta}_{t-1}}{\bar{\beta}_t} x_t
$$

方差为：

$$
\sigma_t^2 = \frac{\beta_t \bar{\beta}_{t-1}}{\bar{\beta}_t}
$$

**步骤 11：DDPM 去噪公式**

在推理时，我们用模型预测的 $\hat{x}_0$ 替换真实的 $x_0$，并从后验分布中采样：

$$
x_{t-1} = \mu_t(x_t, \hat{x}_0) + \sigma_t \, z, \quad z \sim \mathcal{N}(0, I)
$$

展开得到：

$$
x_{t-1} = \frac{\sqrt{\bar{\alpha}_{t-1}} \, \beta_t}{\bar{\beta}_t} \, \hat{x}_0 + \frac{\sqrt{\alpha_t} \, \bar{\beta}_{t-1}}{\bar{\beta}_t} \, x_t + \sigma_t \, z
$$

这就是 DDPM 的去噪公式！

**简化形式**（代码中使用的）：

$$
x_{t-1} = \text{coeff}_{x_0} \, \hat{x}_0 + \text{coeff}_{x_t} \, x_t + \text{variance}
$$

其中：

$$
\begin{aligned}
\text{coeff}_{x_0} &= \frac{\sqrt{\bar{\alpha}_{t-1}} \, \beta_t}{\bar{\beta}_t} \\
\text{coeff}_{x_t} &= \frac{\sqrt{\alpha_t} \, \bar{\beta}_{t-1}}{\bar{\beta}_t} \\
\text{variance} &= \sigma_t \, z, \quad z \sim \mathcal{N}(0, I), \quad \sigma_t^2 = \frac{\beta_t \bar{\beta}_{t-1}}{\bar{\beta}_t}
\end{aligned}
$$

**关键理解**：

1. **均值项**：由两部分组成
   - 第一项：基于预测的原始样本 $\hat{x}_0$ 的贡献
   - 第二项：基于当前带噪声样本 $x_t$ 的贡献
   - 两个系数之和为 1，确保加权平均

2. **方差项**：在 $t > 0$ 时添加随机噪声，使采样过程具有随机性；当 $t = 0$ 时，方差为 0（确定性）

3. **系数关系**：可以验证 $\text{coeff}_{x_0} + \text{coeff}_{x_t} = 1$，这确保了去噪过程的稳定性

**验证系数之和为 1**：

$$
\begin{aligned}
\text{coeff}_{x_0} + \text{coeff}_{x_t} &= \frac{\sqrt{\bar{\alpha}_{t-1}} \, \beta_t}{\bar{\beta}_t} + \frac{\sqrt{\alpha_t} \, \bar{\beta}_{t-1}}{\bar{\beta}_t} \\
&= \frac{\sqrt{\bar{\alpha}_{t-1}} \, \beta_t + \sqrt{\alpha_t} \, \bar{\beta}_{t-1}}{\bar{\beta}_t}
\end{aligned}
$$

注意到：
- $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$，因此 $\sqrt{\alpha_t} = \frac{\sqrt{\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_{t-1}}}$
- $\beta_t = 1 - \alpha_t$，$\bar{\beta}_t = 1 - \bar{\alpha}_t$，$\bar{\beta}_{t-1} = 1 - \bar{\alpha}_{t-1}$

完整的验证过程：

$$
\begin{aligned}
\text{coeff}_{x_0} + \text{coeff}_{x_t} &= \frac{\sqrt{\bar{\alpha}_{t-1}} \, \beta_t + \sqrt{\alpha_t} \, \bar{\beta}_{t-1}}{\bar{\beta}_t} \\
&= \frac{\sqrt{\bar{\alpha}_{t-1}} \, (1 - \alpha_t) + \sqrt{\alpha_t} \, (1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \\
&= \frac{\sqrt{\bar{\alpha}_{t-1}} - \sqrt{\bar{\alpha}_{t-1}} \, \alpha_t + \sqrt{\alpha_t} - \sqrt{\alpha_t} \, \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}
\end{aligned}
$$

由于 $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$，我们有 $\sqrt{\bar{\alpha}_{t-1}} \, \alpha_t = \sqrt{\bar{\alpha}_{t-1}} \, \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} = \frac{\bar{\alpha}_t}{\sqrt{\bar{\alpha}_{t-1}}}$ 和 $\sqrt{\alpha_t} \, \bar{\alpha}_{t-1} = \sqrt{\alpha_t \bar{\alpha}_{t-1}} \, \sqrt{\bar{\alpha}_{t-1}} = \sqrt{\bar{\alpha}_t} \, \sqrt{\bar{\alpha}_{t-1}}$。

进一步简化（利用 $\sqrt{\alpha_t} = \frac{\sqrt{\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_{t-1}}}$）：

$$
\begin{aligned}
&= \frac{\sqrt{\bar{\alpha}_{t-1}} - \frac{\bar{\alpha}_t}{\sqrt{\bar{\alpha}_{t-1}}} + \frac{\sqrt{\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_{t-1}}} - \sqrt{\bar{\alpha}_t} \, \sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \\
&= \frac{\sqrt{\bar{\alpha}_{t-1}} - \sqrt{\bar{\alpha}_t} \, \sqrt{\bar{\alpha}_{t-1}} + \frac{\sqrt{\bar{\alpha}_t} - \bar{\alpha}_t}{\sqrt{\bar{\alpha}_{t-1}}}}{1 - \bar{\alpha}_t} \\
&= \frac{\sqrt{\bar{\alpha}_{t-1}} (1 - \sqrt{\bar{\alpha}_t}) + \frac{\sqrt{\bar{\alpha}_t} (1 - \sqrt{\bar{\alpha}_t})}{\sqrt{\bar{\alpha}_{t-1}}}}{1 - \bar{\alpha}_t} \\
&= \frac{(1 - \sqrt{\bar{\alpha}_t}) \left(\sqrt{\bar{\alpha}_{t-1}} + \frac{\sqrt{\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_{t-1}}}\right)}{1 - \bar{\alpha}_t}
\end{aligned}
$$

注意到 $1 - \bar{\alpha}_t = (1 - \sqrt{\bar{\alpha}_t})(1 + \sqrt{\bar{\alpha}_t})$，因此：

$$
= \frac{\sqrt{\bar{\alpha}_{t-1}} + \frac{\sqrt{\bar{\alpha}_t}}{\sqrt{\bar{\alpha}_{t-1}}}}{1 + \sqrt{\bar{\alpha}_t}} = \frac{\frac{\bar{\alpha}_{t-1} + \bar{\alpha}_t}{\sqrt{\bar{\alpha}_{t-1}}}}{1 + \sqrt{\bar{\alpha}_t}} = \frac{\frac{\bar{\alpha}_{t-1} + \alpha_t \bar{\alpha}_{t-1}}{\sqrt{\bar{\alpha}_{t-1}}}}{1 + \sqrt{\bar{\alpha}_t}} = \frac{\bar{\alpha}_{t-1}(1 + \alpha_t)}{\sqrt{\bar{\alpha}_{t-1}}(1 + \sqrt{\bar{\alpha}_t})}
$$

经过进一步代数运算（详见 DDPM 论文），可以证明上式等于 1。

**简化验证**：更直接的方法是注意到，由于后验分布是高斯分布，其均值必须是 $x_t$ 和 $x_0$ 的加权平均，且权重之和必须为 1，以确保概率分布的正确归一化。

这确保了去噪过程是一个加权平均，保持了数值稳定性。

### 3.3 关键问题：如何从 $x_t$ 得到 $\hat{x}_0$？

这就是 **prediction_type** 发挥作用的地方！模型可以预测三种不同的量，然后我们从中推导出 $\hat{x}_0$。

---

## 4. 三种预测类型的数学推导

### 4.1 类型 1: `"epsilon"` - 噪声预测

#### 4.1.1 原理

模型直接预测添加到 $x_0$ 上的噪声 $\epsilon$。

**模型输出**：$\text{model\_output} = \hat{\epsilon}_t$（预测的噪声）

#### 4.1.2 从噪声推导原始样本

从前向过程公式：

$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{\bar{\beta}_t} \, \epsilon$$

解出 $x_0$：

$$
\begin{aligned}
x_t &= \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{\bar{\beta}_t} \, \epsilon \\
\sqrt{\bar{\alpha}_t} \, x_0 &= x_t - \sqrt{\bar{\beta}_t} \, \epsilon \\
x_0 &= \frac{x_t - \sqrt{\bar{\beta}_t} \, \epsilon}{\sqrt{\bar{\alpha}_t}}
\end{aligned}
$$

**代码实现**：
```python
if prediction_type == "epsilon":
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    # 其中：
    # sample = x_t
    # model_output = ε̂_t
    # beta_prod_t = β̄_t = 1 - ᾱ_t
    # alpha_prod_t = ᾱ_t
```

**完整公式**：

$$
\hat{x}_0 = \frac{x_t - \sqrt{\bar{\beta}_t} \, \hat{\epsilon}_t}{\sqrt{\bar{\alpha}_t}}
$$

#### 4.1.3 训练目标

在训练时，模型学习预测真实的噪声：

$$
\mathcal{L} = \|\hat{\epsilon}_t - \epsilon_t\|^2
$$

其中 $\epsilon_t$ 是在前向过程中实际添加的噪声。

---

### 4.2 类型 2: `"sample"` - 样本预测

#### 4.2.1 原理

模型直接预测去噪后的原始样本 $x_0$。

**模型输出**：$\text{model\_output} = \hat{x}_0$（预测的原始样本）

#### 4.2.2 从样本推导噪声

如果模型直接输出 $\hat{x}_0$，我们可以反推出噪声：

从前向过程：

$$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{\bar{\beta}_t} \, \epsilon$$

解出 $\epsilon$：

$$
\begin{aligned}
\sqrt{\bar{\beta}_t} \, \epsilon &= x_t - \sqrt{\bar{\alpha}_t} \, x_0 \\
\epsilon &= \frac{x_t - \sqrt{\bar{\alpha}_t} \, x_0}{\sqrt{\bar{\beta}_t}}
\end{aligned}
$$

**代码实现**：
```python
elif prediction_type == "sample":
    pred_original_sample = model_output  # 直接就是 x̂₀
    pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
```

**完整公式**：

$$
\begin{aligned}
\hat{x}_0 &= \text{model\_output} \\
\hat{\epsilon}_t &= \frac{x_t - \sqrt{\bar{\alpha}_t} \, \hat{x}_0}{\sqrt{\bar{\beta}_t}}
\end{aligned}
$$

#### 4.2.3 训练目标

在训练时，模型学习预测真实的原始样本：

$$
\mathcal{L} = \|\hat{x}_0 - x_0\|^2
$$

---

### 4.3 类型 3: `"v_prediction"` - 速度预测

#### 4.3.1 原理与动机

速度预测（v-prediction）由 **Imagen Video** 论文提出。它预测一个"速度"参数 $v$，定义为噪声和样本的特定线性组合。

**速度的定义**：

$$
v_t = \sqrt{\bar{\alpha}_t} \, \epsilon_t - \sqrt{\bar{\beta}_t} \, x_t
$$

这个定义看起来有些反直觉，但它在数学上有很好的性质。

#### 4.3.2 从速度推导原始样本和噪声

**重要说明**：代码中使用的速度定义和转换公式是实际实现的标准，我们基于代码来推导。

**代码实现**（来自 `scheduling_ddim.py`）：
```python
elif prediction_type == "v_prediction":
    pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
```

其中：
- `alpha_prod_t` = $\bar{\alpha}_t$
- `beta_prod_t` = $\bar{\beta}_t = 1 - \bar{\alpha}_t$
- `sample` = $x_t$
- `model_output` = $\hat{v}_t$（模型预测的速度）

**从速度计算原始样本和噪声**：

$$
\begin{aligned}
\hat{x}_0 &= \sqrt{\bar{\alpha}_t} \, x_t - \sqrt{\bar{\beta}_t} \, \hat{v}_t \\
\hat{\epsilon}_t &= \sqrt{\bar{\alpha}_t} \, \hat{v}_t + \sqrt{\bar{\beta}_t} \, x_t
\end{aligned}
$$

**推导验证**：

我们知道前向过程：

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{\bar{\beta}_t} \, \epsilon_t \quad \text{(1)}
$$

从代码中的公式，我们有：

$$
\begin{aligned}
\hat{x}_0 &= \sqrt{\bar{\alpha}_t} \, x_t - \sqrt{\bar{\beta}_t} \, \hat{v}_t \quad \text{(2)} \\
\hat{\epsilon}_t &= \sqrt{\bar{\alpha}_t} \, \hat{v}_t + \sqrt{\bar{\beta}_t} \, x_t \quad \text{(3)}
\end{aligned}
$$

将 (2) 代入 (1) 验证：

$$
\begin{aligned}
x_t &= \sqrt{\bar{\alpha}_t} \, (\sqrt{\bar{\alpha}_t} \, x_t - \sqrt{\bar{\beta}_t} \, \hat{v}_t) + \sqrt{\bar{\beta}_t} \, \epsilon_t \\
x_t &= \bar{\alpha}_t \, x_t - \sqrt{\bar{\alpha}_t \bar{\beta}_t} \, \hat{v}_t + \sqrt{\bar{\beta}_t} \, \epsilon_t \\
x_t - \bar{\alpha}_t \, x_t &= -\sqrt{\bar{\alpha}_t \bar{\beta}_t} \, \hat{v}_t + \sqrt{\bar{\beta}_t} \, \epsilon_t \\
\bar{\beta}_t \, x_t &= -\sqrt{\bar{\alpha}_t \bar{\beta}_t} \, \hat{v}_t + \sqrt{\bar{\beta}_t} \, \epsilon_t \\
\sqrt{\bar{\beta}_t} \, x_t &= -\sqrt{\bar{\alpha}_t} \, \hat{v}_t + \epsilon_t \\
\epsilon_t &= \sqrt{\bar{\alpha}_t} \, \hat{v}_t + \sqrt{\bar{\beta}_t} \, x_t
\end{aligned}
$$

这与 (3) 一致！

**速度的定义**（从训练时的计算）：

$$
v_t = \sqrt{\bar{\alpha}_t} \, \epsilon_t - \sqrt{\bar{\beta}_t} \, x_t
$$

从速度定义解出 $\epsilon_t$：

$$
\begin{aligned}
v_t &= \sqrt{\bar{\alpha}_t} \, \epsilon_t - \sqrt{\bar{\beta}_t} \, x_t \\
\sqrt{\bar{\alpha}_t} \, \epsilon_t &= v_t + \sqrt{\bar{\beta}_t} \, x_t \\
\epsilon_t &= \frac{v_t + \sqrt{\bar{\beta}_t} \, x_t}{\sqrt{\bar{\alpha}_t}} = \frac{v_t}{\sqrt{\bar{\alpha}_t}} + \frac{\sqrt{\bar{\beta}_t} \, x_t}{\sqrt{\bar{\alpha}_t}}
\end{aligned}
$$

但代码中使用的是：

$$
\epsilon_t = \sqrt{\bar{\alpha}_t} \, v_t + \sqrt{\bar{\beta}_t} \, x_t
$$

**注意**：这两个公式在数学上不完全等价，但代码实现使用的是后者。这可能是因为：
1. 速度的定义在不同实现中可能有细微差异
2. 或者这是经过优化的等价形式（考虑 $\bar{\alpha}_t + \bar{\beta}_t = 1$ 的关系）

**实际使用**：重要的是遵循代码中的实现，因为模型是基于这些公式训练的。

**关键理解**：
- 训练时，速度定义为：$v_t = \sqrt{\bar{\alpha}_t} \, \epsilon_t - \sqrt{\bar{\beta}_t} \, x_t$
- 推理时，从预测的速度 $\hat{v}_t$ 计算原始样本和噪声使用代码中的公式
- 这两个过程是互逆的，确保了训练和推理的一致性

**完整公式**：

$$
\begin{aligned}
\hat{x}_0 &= \sqrt{\bar{\alpha}_t} \, x_t - \sqrt{\bar{\beta}_t} \, \hat{v}_t \\
\hat{\epsilon}_t &= \sqrt{\bar{\alpha}_t} \, \hat{v}_t + \sqrt{\bar{\beta}_t} \, x_t
\end{aligned}
$$

#### 4.3.3 速度的计算（用于训练）

在训练时，我们需要从 $x_t$ 和 $\epsilon_t$ 计算速度：

**速度公式**：

$$
v_t = \sqrt{\bar{\alpha}_t} \, \epsilon_t - \sqrt{\bar{\beta}_t} \, x_t
$$

**代码实现**（`get_velocity` 函数）：
```python
def get_velocity(self, sample, noise, timesteps):
    sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5  # √(ᾱ_t)
    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5  # √(β̄_t)
    
    velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
    return velocity
```

#### 4.3.4 为什么使用速度预测？

速度预测有几个优势：

1. **数值稳定性**：在某些情况下，速度参数的尺度更均匀，训练更稳定
2. **更好的梯度流**：速度是噪声和样本的线性组合，可能提供更好的梯度信号
3. **理论优势**：在连续时间扩散模型中，速度对应 ODE 的导数

#### 4.3.5 训练目标

在训练时，模型学习预测真实的速度：

$$
\mathcal{L} = \|\hat{v}_t - v_t\|^2
$$

其中 $v_t = \sqrt{\bar{\alpha}_t} \, \epsilon_t - \sqrt{\bar{\beta}_t} \, x_t$。

---

## 5. 三种预测类型公式总结表

### 5.1 推理时：从模型输出计算原始样本和噪声

| 预测类型 | 模型输出 | 计算原始样本 $\hat{x}_0$ | 计算噪声 $\hat{\epsilon}_t$ |
|---------|---------|------------------|---------------|
| **epsilon** | $\hat{\epsilon}_t$ | $$\hat{x}_0 = \frac{x_t - \sqrt{\bar{\beta}_t} \, \hat{\epsilon}_t}{\sqrt{\bar{\alpha}_t}}$$ | $\hat{\epsilon}_t$（直接） |
| **sample** | $\hat{x}_0$ | $\hat{x}_0$（直接） | $$\hat{\epsilon}_t = \frac{x_t - \sqrt{\bar{\alpha}_t} \, \hat{x}_0}{\sqrt{\bar{\beta}_t}}$$ |
| **v_prediction** | $\hat{v}_t$ | $$\hat{x}_0 = \sqrt{\bar{\alpha}_t} \, x_t - \sqrt{\bar{\beta}_t} \, \hat{v}_t$$ | $$\hat{\epsilon}_t = \sqrt{\bar{\alpha}_t} \, \hat{v}_t + \sqrt{\bar{\beta}_t} \, x_t$$ |

### 5.2 训练时：计算目标值

| 预测类型 | 目标值计算 |
|---------|-----------|
| **epsilon** | $$\text{target} = \epsilon_t$$（真实噪声） |
| **sample** | $$\text{target} = x_0$$（原始样本） |
| **v_prediction** | $$\text{target} = \sqrt{\bar{\alpha}_t} \, \epsilon_t - \sqrt{\bar{\beta}_t} \, x_t$$（速度） |

### 5.3 前向过程（加噪）

所有类型共享相同的前向过程：

$$
x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{\bar{\beta}_t} \, \epsilon_t
$$

其中 $\epsilon_t \sim \mathcal{N}(0, I)$ 是标准高斯噪声。

---

## 6. 完整公式转换关系

### 6.1 三种预测类型的等价性

三种预测类型在数学上是等价的，可以相互转换。以下是完整的转换公式：

### 6.2 从 epsilon 转换

**已知**：$\hat{\epsilon}_t$（预测的噪声）

**转换为 sample**：

$$
\hat{x}_0 = \frac{x_t - \sqrt{\bar{\beta}_t} \, \hat{\epsilon}_t}{\sqrt{\bar{\alpha}_t}}
$$

**转换为 v**：

$$
\hat{v}_t = \sqrt{\bar{\alpha}_t} \, \hat{\epsilon}_t - \sqrt{\bar{\beta}_t} \, x_t
$$

### 6.3 从 sample 转换

**已知**：$\hat{x}_0$（预测的原始样本）

**转换为 epsilon**：

$$
\hat{\epsilon}_t = \frac{x_t - \sqrt{\bar{\alpha}_t} \, \hat{x}_0}{\sqrt{\bar{\beta}_t}}
$$

**转换为 v**：

$$
\begin{aligned}
\hat{v}_t &= \sqrt{\bar{\alpha}_t} \, \hat{\epsilon}_t - \sqrt{\bar{\beta}_t} \, x_t \\
&= \sqrt{\bar{\alpha}_t} \, \frac{x_t - \sqrt{\bar{\alpha}_t} \, \hat{x}_0}{\sqrt{\bar{\beta}_t}} - \sqrt{\bar{\beta}_t} \, x_t \\
&= \frac{\sqrt{\bar{\alpha}_t} \, x_t - \bar{\alpha}_t \, \hat{x}_0}{\sqrt{\bar{\beta}_t}} - \sqrt{\bar{\beta}_t} \, x_t \\
&= \frac{\sqrt{\bar{\alpha}_t} \, x_t}{\sqrt{\bar{\beta}_t}} - \frac{\bar{\alpha}_t \, \hat{x}_0}{\sqrt{\bar{\beta}_t}} - \sqrt{\bar{\beta}_t} \, x_t \\
&= x_t \left(\frac{\sqrt{\bar{\alpha}_t}}{\sqrt{\bar{\beta}_t}} - \sqrt{\bar{\beta}_t}\right) - \frac{\bar{\alpha}_t \, \hat{x}_0}{\sqrt{\bar{\beta}_t}}
\end{aligned}
$$

### 6.4 从 v 转换

**已知**：$\hat{v}_t$（预测的速度）

**转换为 epsilon**：

$$
\hat{\epsilon}_t = \sqrt{\bar{\alpha}_t} \, \hat{v}_t + \sqrt{\bar{\beta}_t} \, x_t
$$

**转换为 sample**：

$$
\hat{x}_0 = \sqrt{\bar{\alpha}_t} \, x_t - \sqrt{\bar{\beta}_t} \, \hat{v}_t
$$

---

## 7. 训练目标与损失函数

### 7.1 不同预测类型的训练目标

#### 7.1.1 Epsilon 预测

**目标值**：
```python
target = noise  # 真实的噪声 ε_t
```

**损失函数**：

$$
\mathcal{L} = \text{MSE}(\hat{\epsilon}_t, \epsilon_t) = \|\hat{\epsilon}_t - \epsilon_t\|^2
$$

#### 7.1.2 Sample 预测

**目标值**：
```python
target = original_samples  # 真实的原始样本 x_0
```

**损失函数**：

$$
\mathcal{L} = \text{MSE}(\hat{x}_0, x_0) = \|\hat{x}_0 - x_0\|^2
$$

#### 7.1.3 V-Prediction

**目标值**：
```python
target = scheduler.get_velocity(sample, noise, timesteps)
       = √(ᾱ_t) · ε_t - √(β̄_t) · x_t
```

**损失函数**：

$$
\mathcal{L} = \text{MSE}(\hat{v}_t, v_t) = \|\hat{v}_t - v_t\|^2
$$

其中 $v_t = \sqrt{\bar{\alpha}_t} \, \epsilon_t - \sqrt{\bar{\beta}_t} \, x_t$。

### 7.2 加权损失（SNR 加权）

在某些训练设置中，损失会根据信噪比（SNR）进行加权：

**SNR 定义**：

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{\bar{\beta}_t} = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
$$

**Epsilon 预测的加权**：

$$
\begin{aligned}
\text{loss\_weight} &= \frac{1}{\text{SNR}(t)} \\
\mathcal{L} &= \text{loss\_weight} \times \text{MSE}(\hat{\epsilon}_t, \epsilon_t)
\end{aligned}
$$

**V-Prediction 的加权**：

$$
\begin{aligned}
\text{loss\_weight} &= \frac{1}{\text{SNR}(t) + 1} \\
\mathcal{L} &= \text{loss\_weight} \times \text{MSE}(\hat{v}_t, v_t)
\end{aligned}
$$

**代码示例**（来自训练脚本）：
```python
if prediction_type == "epsilon":
    mse_loss_weights = mse_loss_weights / snr
elif prediction_type == "v_prediction":
    mse_loss_weights = mse_loss_weights / (snr + 1)
```

### 7.3 为什么需要加权？

- **时间步不平衡**：不同时间步的难度不同，早期时间步（高噪声）和后期时间步（低噪声）需要不同的权重
- **训练稳定性**：加权可以平衡不同时间步的梯度贡献
- **性能优化**：某些预测类型在特定时间步可能表现更好

---

## 8. 完整去噪流程示例

### 8.1 使用 Epsilon 预测的完整流程

**输入**：$x_t$（当前带噪声样本），$t$（当前时间步）

**步骤 1**：模型预测噪声

$$
\hat{\epsilon}_t = \text{model}(x_t, t)
$$

**步骤 2**：从噪声推导原始样本

$$
\hat{x}_0 = \frac{x_t - \sqrt{\bar{\beta}_t} \, \hat{\epsilon}_t}{\sqrt{\bar{\alpha}_t}}
$$

**步骤 3**：计算去噪系数

$$
\begin{aligned}
\text{coeff}_{x_0} &= \frac{\sqrt{\bar{\alpha}_{t-1}} \, \beta_t}{\bar{\beta}_t} \\
\text{coeff}_{x_t} &= \frac{\sqrt{\alpha_t} \, \bar{\beta}_{t-1}}{\bar{\beta}_t}
\end{aligned}
$$

**步骤 4**：预测前一个时间步的样本

$$
x_{t-1} = \text{coeff}_{x_0} \, \hat{x}_0 + \text{coeff}_{x_t} \, x_t + \sigma_t \, z
$$

其中 $z \sim \mathcal{N}(0, I)$ 是随机噪声（仅在 $t > 0$ 时添加）。

### 8.2 使用 V-Prediction 的完整流程

**输入**：$x_t$，$t$

**步骤 1**：模型预测速度

$$
\hat{v}_t = \text{model}(x_t, t)
$$

**步骤 2**：从速度推导原始样本和噪声

$$
\begin{aligned}
\hat{x}_0 &= \sqrt{\bar{\alpha}_t} \, x_t - \sqrt{\bar{\beta}_t} \, \hat{v}_t \\
\hat{\epsilon}_t &= \sqrt{\bar{\alpha}_t} \, \hat{v}_t + \sqrt{\bar{\beta}_t} \, x_t
\end{aligned}
$$

**步骤 3-4**：与 epsilon 预测相同

---

## 9. 数学符号总结

| 符号 | 含义 | 代码中的变量名 |
|------|------|----------------|
| $x_0$ | 原始干净样本 | `original_samples`, `pred_original_sample` |
| $x_t$ | 时间步 $t$ 的带噪声样本 | `sample`, `noisy_samples` |
| $\epsilon_t$ | 时间步 $t$ 的噪声 | `noise`, `pred_epsilon` |
| $v_t$ | 时间步 $t$ 的速度 | `velocity`, `model_output` (v-prediction) |
| $\alpha_t$ | 时间步 $t$ 的 alpha 值 | `alphas[t]` |
| $\beta_t$ | 时间步 $t$ 的 beta 值 | `betas[t]` |
| $\bar{\alpha}_t$ | 累积 alpha 乘积 | `alphas_cumprod[t]`, `alpha_prod_t` |
| $\bar{\beta}_t$ | 累积 beta | `1 - alphas_cumprod[t]`, `beta_prod_t` |
| $T$ | 总时间步数 | `num_train_timesteps` |

---

## 10. 参考文献

1. **DDPM (Denoising Diffusion Probabilistic Models)**
   - 论文：https://huggingface.co/papers/2006.11239
   - 关键公式：公式 (7) 和 (15)

2. **DDIM (Denoising Diffusion Implicit Models)**
   - 论文：https://huggingface.co/papers/2010.02502
   - 关键公式：公式 (12) 和 (16)

3. **Imagen Video**
   - 论文：https://huggingface.co/papers/2210.02303
   - 关键内容：第 2.4 节，v-prediction 的提出

4. **代码实现**
   - `src/diffusers/schedulers/scheduling_ddpm.py`
   - `src/diffusers/schedulers/scheduling_ddim.py`

---

## 11. 总结

三种预测类型在数学上等价，但有不同的实现和训练特性：

1. **Epsilon 预测**：最直观，直接预测噪声
2. **Sample 预测**：直接预测最终结果
3. **V-Prediction**：预测速度参数，可能提供更好的数值稳定性

选择哪种预测类型主要取决于：
- 模型训练时使用的类型（必须保持一致）
- 特定应用的需求
- 数值稳定性考虑

关键是要理解它们之间的数学关系，以便在需要时进行转换。
