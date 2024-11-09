![image](https://github.com/user-attachments/assets/bdb3dba7-779d-4bf3-944d-d5f94eaea30a)# SRRFN_paddle

Lightweight and Accurate Recursive Fractal Network for Image Super-Resolution

[Paper]
(http://openaccess.thecvf.com/content_ICCVW_2019/html/LCI/Li_Lightweight_and_Accurate_Recursive_Fractal_Network_for_Image_Super-Resolution_ICCVW_2019_paper.html)

Paddle 复现版本

## 数据集
DIV2K
https://data.vision.ee.ethz.ch/cvl/DIV2K/
Set5
https://drive.google.com/drive/folders/1pRmhEmmY-tPF7uH8DuVthfHoApZWJ1QU


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
