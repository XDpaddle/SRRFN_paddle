# SRRFN_paddle

ClassSR: Lightweight and Accurate Recursive Fractal Network for Image Super-Resolution

[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Kong_ClassSR_A_General_Framework_to_Accelerate_Super-Resolution_Networks_by_Data_CVPR_2021_paper.pdf)


Paddle 复现版本

## 数据集

分类之后训练集用于训练SR模块
https://aistudio.baidu.com/aistudio/datasetdetail/106261
## aistudio
脚本任务地址: https://aistudio.baidu.com/aistudio/clusterprojectdetail/2356381
## 训练模型
链接：https://pan.baidu.com/s/1AhuC4hRTfXdpP93Tqs2NLg 
提取码：1234 

## 训练步骤
### train sr
```bash
python train.py -opt config/train/train_SRRFN.yml
```


## 测试步骤
```bash
python test.py -opt config/test/test_SRRFN.yml
```
## 参考资料

- [Xiangtaokong/ClassSR](https://github.com/Xiangtaokong/ClassSR)