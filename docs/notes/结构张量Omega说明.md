# 结构张量公式中 $\Omega(i)$ 的范围

在公式

$$
J_{22}(i)=\sum_{j\in\Omega(i)}w(i,j)G_y(j)^2
$$

以及

$$
J_{12}(i)=J_{21}(i)=\sum_{j\in\Omega(i)}w(i,j)G_x(j)G_y(j)
$$

中，$\Omega(i)$ 表示以像素 $i$ 为中心、用于计算结构张量的局部邻域。仅根据公式本身无法确定它的具体大小，还需要结合权重函数 $w(i,j)$ 的定义。

在当前项目代码中，$w(i,j)$ 是高斯权重，因此 $\Omega(i)$ 是以 $i$ 为中心的方形高斯窗口：

$$
\Omega(i)=
\left\{
j:\left|x_j-x_i\right|\leq3\sigma,
\left|y_j-y_i\right|\leq3\sigma
\right\}.
$$

窗口尺寸为：

$$
(6\sigma+1)\times(6\sigma+1),
$$

其中：

$$
\sigma=\max\left(3,
\left\lfloor\frac{\min(H,W)}{100}\right\rfloor
\right).
$$

$H$ 和 $W$ 分别表示图像高度和宽度。

严格考虑图像边界时，可以写成：

$$
\Omega(i)=
\left(
[x_i-3\sigma,x_i+3\sigma]
\times
[y_i-3\sigma,y_i+3\sigma]
\right)\cap\mathcal I,
$$

其中 $\mathcal I$ 表示图像定义域。

## 示例

| 图像短边 $\min(H,W)$ | $\sigma$ | $\Omega(i)$ 的窗口尺寸 |
|---:|---:|---:|
| 256 | 3 | $19\times19$ |
| 920 | 9 | $55\times55$ |
| 1200 | 12 | $73\times73$ |

## 有效掩膜的影响

项目只在有效分析掩膜内保留 Scharr 梯度。掩膜外的 $G_x(j)$ 和 $G_y(j)$ 被置为 0，然后再对结构张量分量进行高斯平滑。因此，$\Omega(i)$ 的几何范围仍然是上述方形窗口，但真正提供纹理梯度信息的是窗口内属于有效掩膜的像素。

需要注意，结构张量中的 $\Omega(i)$ 是高斯加权方形邻域，不是 Severity Map 中由 Heatmap Radius 定义的圆形邻域。
