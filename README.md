# 背景建模与运动目标检测

## 帧间差法目标检测

帧间差法（Frame Difference Method）是一种视频目标检测的基本技术，它利用相邻帧之间的差异来检测运动目标。

帧间差法的核心原理是检测相邻帧之间的差异，其基本步骤如下：

1. **帧差计算**：对于每一对相邻帧，计算它们之间每个像素的差异。这可以通过简单地逐像素相减来实现。
2. **阈值处理**：对帧差图像应用阈值处理，将像素差异分为前景（运动目标）和背景两类。像素差异大于阈值的像素被标记为前景，而像素差异小于阈值的像素被标记为背景。
3. **目标检测**：对阈值化后的帧差图像进行连通分量分析或轮廓检测，以检测出运动目标的位置和轮廓。

帧间差法可以用以下公式表示：

$$
I_{\text{diff}}(x, y, t) = |I(x, y, t) - I(x, y, t-1)|
$$

其中：

- $I_{\text{diff}}(x, y, t)$ 是帧差图像的像素值。
- $I(x, y, t)$ 是当前帧在坐标 \((x, y)\) 处的像素值。
- $I(x, y, t-1)$ 是前一帧在相同坐标处的像素值。

## 单高斯背景建模

1. **背景建模**：首先，需要建立背景模型。采用初始帧初始化背景。
2. **高斯分布建模**：建立背景模型时，通常使用单一高斯分布来建模背景像素的颜色分布。这个高斯分布由两个关键参数组成：

   - 均值（mean）：表示背景颜色的平均值。
   - 标准差（standard deviation）：表示颜色值在背景中的变化程度。

   高斯分布的概率密度函数：

   $$
   f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \cdot e^{-\frac{(x - \mu)^2}{2\sigma^2}}
   $$

   使用一段时间T内的视频帧初始化图像均值和标准差，这里使用12帧图像。
3. **差异检测**：在每一帧中，将当前像素的颜色值与背景模型的高斯分布进行比较。如果像素值与高斯分布相符，将其标记为背景；如果不符，将其标记为前景。通常，可以使用阈值来确定何时将像素标记为前景。如果像素值与高斯分布差异超过阈值，则被认为是前景。如果

   $$
   |I(x,y) - \mu_{t-1}(x,y)| < k \cdot \sigma_{t-1}(x,y)
   $$

   那么将该像素点(x,y)就是背景点，进行参数更新。

   $$
   \mu_{t}(x,y) = (1-\lambda)*\mu_{t-1}(x,y)+\lambda*I(x,y)\\
   \sigma_{t}^{2}(x,y) = (1-\lambda)*\sigma_{t-1}^{2}(x,y) + \lambda * \
            (I(x,y) - \mu_{t}(x,y))^{2}
   $$

   进行背景更新

   $$
   B_{t}(x,y) = (1-\lambda)*B_{t-1}(x,y)+\lambda*I(x,y)
   $$
4. **目标检测**：通过差异检测，可以轻松检测出前景目标。不符合背景模型的像素被标记为前景，这有助于定位运动的目标对象。**结果储存于'output/'**

## 混合高斯模型
