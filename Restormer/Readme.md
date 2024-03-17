# Restormer:Efficient Transformer for High-Resolution Image Restoration(CVPR 2022 -- Oral)
**Restormer: Efficient Transformer for High-Resolution Image Restoration**  
论文地址：[Restormer: Efficient Transformer for High-Resolution Image Restoration](https://arxiv.org/abs/2111.09881)  
源码：[Restormer](https://github.com/swz30/Restormer)  
## 简介
![image](https://github.com/ZzhuL/DeepL_CV/assets/83905469/daab4b40-424e-4382-b8eb-341fa1cbf6bf)

**Contribution:**  
* 提出了Restormer，一种编码-解码结构的Transformer，用于在高分辨率图像上进行多尺度局部/全局表示学习，而不将它们分解成局部窗口，从而利用遥远的图像上下文。  
* 提出了一种使用深度卷积的多头转置注意模块(multi-Dconv head transposed attention, MDTA)，它能够聚合局部和非局部像素交互，并且足够有效地处理高分辨率图像。  
* 一种新的使用深度卷积的门控前馈网络(Gated-Dconv feed-forward network, GDFN)，它执行受控的特征转换，即抑制信息量较少的特征，只允许有用的信息进一步通过网络层次结构。  


# 代码复现
## 1.配置
* GPU RTX 3080 Ti(12GB) * 1
* CPU 12 vCPU Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz
* PyTorch  1.8.1
* Python  3.8(ubuntu18.04)
* Cuda  11.1

