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
## 2.数据集
数据集后续介绍

## 3.源码问题
我在autodl上租的GPU进行运行，除了按照INSTALL中安装一些必备库，仍有问题不能完全运行代码。我主要遇到的问题与解决方案主要如下：
### 3.1 修改配置文件
修改Deraining/Options/Deraining_Restormer.py中GPU数量，按需修改。如果只有一个GPU，则用不到分布式训练，train.sh脚本需要修改。
```
#!/usr/bin/env bash

# CONFIG=$1

export NCCL_P2P_DISABLE=1

python setup.py develop --no_cuda_ext

CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt Options/Deraining.yml
```
CUDA_VISIBLE_DEVICES=0，指的是使用GPU0，原论文中使用了4个GPU，CUDA_VISIBLE_DEVICES=0,1,2,3 分别对应GPU0,GPU1,GPU2,GPU3。
### 3.2 No module named 'basicsr'
参考这个[解决方案](https://blog.csdn.net/G_B_L/article/details/106745534)
```
  File "basicsr/train.py", line 10, in <module>
    from basicsr.data import create_dataloader, create_dataset
ImportError: cannot import name 'create_dataloader' from 'basicsr.data' (/root/miniconda3/lib/python3.8/site-packages/basicsr/data/__init__.py)
```
解决方法：为python解释器指定搜索路径,即把basicsr的路径添加到环境变量里。可以在train.py文件的开始加入以下代码，并使得basicsr在root_path路径下。在'from basicsr.data import create_dataloader, create_dataset'前加入下段代码：
```
import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)
```
#### 后来发现是cuda版本问题
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
### 3.3 缺少python包
根据我添加的`requirements.txt`文件，run
```
pip install -r requirements.txt
```
## 4.CPU test
在拥有预训练模型的前提下，复现代码想节约成本，在自己电脑上进行测试，可能会遇到显存不足的情况，因此要对图像进行切分，分部处理。首先加入以下两行代码
```
parser.add_argument('--tile', type=int, default=80, help='Tile size (e.g 720). None means testing on the original resolution image')
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
```
再加入以下对图片进行切片处理的代码
```
            if args.tile is None:
                print("tile is None")
                restored = model_restoration(input_)
            else:
                # test the image tile by tile
                b, c, h, w = input_.shape
                tile = min(args.tile, h, w)
                print(tile)
                assert tile % 8 == 0, "tile size should be multiple of 8"
                tile_overlap = args.tile_overlap

                stride = tile - tile_overlap
                h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
                w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
                E = torch.zeros(b, c, h, w).type_as(input_)
                W = torch.zeros_like(E)

                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                        out_patch = model_restoration(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)

                        E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                        W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
                restored = E.div_(W)
            # restored = model_restoration(input_)
            restored = torch.clamp(restored, 0, 1)
```
```
project
│   README.md
│   file001.txt    
│
└───folder1
│   │   file011.txt
│   │   file012.txt
│   │
│   └───subfolder1
│       │   file111.txt
│       │   file112.txt
│       │   ...
│   
└───folder2
    │   file021.txt
    │   file022.txt
```
