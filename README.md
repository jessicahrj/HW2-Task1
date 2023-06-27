# HW2-Task1
HW2-Task1

# Project1项目概述：
  使用CNN网络模型(自己设计或使用现有的CNN架构，如AlexNet，ResNet-18)作为baseline在CIFAR-100上训练并测试；对比cutmix, cutout, mixup三种方法以及baseline方法在CIFAR-100图像分类任务中的性能表现；对三张训练样本分别经过cutmix, cutout, mixup后进行可视化，一共show 9张图像。
  
# 项目架构
1. 采用ResNet50对CIFAR-100图像进行分类。resnet.py放在models文件夹中。
2. 对数据集进行data augmentation：cutmix.py, cutout.py, mixup.py放在data_aug文件夹中。
3. dataset.py文件用来读取CIFAR-10；picture.py选择数据增强方式处理样本；train,py和test.py训练和测试。
4. 参数设置： batch_size=128；learning_rate初始化=0.1；learning_rate decay: MILESTONES=[60,120,160];γ=0.2 warmup从0到0.1 in first epoch; 优化器：momentum SGD with momentum=0.9 and decay_rate=5e-4；EPOCH=200; 损失函数：Cross Entropy Loss。

# 模型参数保存
百度网盘链接：https://pan.baidu.com/s/1kMe1JOqYZa8x9qehgokVRg 
提取码：stnz
