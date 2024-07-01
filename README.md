ACNET
===========================
This is the official implement for
<div class="highlight highlight-html"><pre>
<b><a href=https://arxiv.org/abs/1905.10089>ACNET: ATTENTION BASED NETWORK TO EXPLOIT COMPLEMENTARY FEATURES FOR RGBD SEMANTIC SEGMENTATION</a>    
<a href=https://github.com/anheidelonghu>Xinxin Hu*</a>, <a href=http://www.yangkailun.com/>Kailun Yang</a>, Lei Fei, <a href=http://wangkaiwei.org/blog.html> Kaiwei Wang</a></b>
Accepted by IEEE ICIP 2019
</pre></div>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/acnet-attention-based-network-to-exploit/semantic-segmentation-on-sun-rgbd)](https://paperswithcode.com/sota/semantic-segmentation-on-sun-rgbd?p=acnet-attention-based-network-to-exploit)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/acnet-attention-based-network-to-exploit/semantic-segmentation-on-nyu-depth-v2)](https://paperswithcode.com/sota/semantic-segmentation-on-nyu-depth-v2?p=acnet-attention-based-network-to-exploit)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/acnet-attention-based-network-to-exploit/thermal-image-segmentation-on-pst900)](https://paperswithcode.com/sota/thermal-image-segmentation-on-pst900?p=acnet-attention-based-network-to-exploit)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/acnet-attention-based-network-to-exploit/thermal-image-segmentation-on-mfn-dataset)](https://paperswithcode.com/sota/thermal-image-segmentation-on-mfn-dataset?p=acnet-attention-based-network-to-exploit)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/acnet-attention-based-network-to-exploit/semantic-segmentation-on-kitti-360)](https://paperswithcode.com/sota/semantic-segmentation-on-kitti-360?p=acnet-attention-based-network-to-exploit)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/acnet-attention-based-network-to-exploit/semantic-segmentation-on-thud-robotic-dataset)](https://paperswithcode.com/sota/semantic-segmentation-on-thud-robotic-dataset?p=acnet-attention-based-network-to-exploit)

## Experiment result
we evaluate the mIoU of ACNet in SUN-RGBD and NYUDv2

|    | SUN-RGBD | NYUDv2 |
|----|----|----|
| mIoU | 48.1% | 48.3% |


## How to use
This code is NYUDv2 implement.

### Requestments
```
Python 3
Pytorch 0.4.1
TensorboardX
Tensorboard
```

### Train the mode

```
python ACNet_train_V1_nyuv2.py --cuda -b 4
```

### Evaluate the mode
The pretrained model for NYUDv2 is avaliable at http://wangkaiwei.org/file/NYUDv2.zip .

The pre-processed NYUDv2 dataset with .npy format is avaliable at https://pan.baidu.com/s/1nxys4pdT4pacWLMScIGT4w  with password：k6rv 

```
python ACNet_eval_nyuv2.py --cuda --last-ckpt '.\model\ckpt_epoch_1195.00.pth'
```

*Note: The code is partially based on RedNet (https://github.com/JindongJiang/RedNet) and (https://github.com/warmspringwinds/pytorch-segmentation-detection)*

