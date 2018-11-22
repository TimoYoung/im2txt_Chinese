# im2txt_Chinese
深度学习实现图像中文描述项目实践
## 简介
基于tensorflow的深度学习项目。实现了输入一张图片自动生成基于图片内容的自然语言描述。   
代码主体来自于Tensorflow Model repository中的**[im2txt](https://github.com/tensorflow/models/tree/master/research/im2txt)**项目。 本项目没有改动任何模型算法，只是在生成语言描述部分进行中文复现。 
数据集采用AI Challenger **[图像中文描述数据集](https://challenger.ai/dataset/caption)**
## 模型介绍
本项目采用了**Show and Tell**模型。见于论文：  
["Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge."](http://arxiv.org/abs/1609.06647)

其主要思想为将训练过程分为编码与解码两部分，首先将一张图片编码成为一组定长的向量表示，再根据向量化的图像特征解码为自然语言输出。在编码过程中，使用深度卷积神经网络**Inception v3**图像识别模型。解码采用循环神经网络 **LSTM** 来进行语言建模。

### 模型结构
![Show and Tell model](doc_image/show_and_tell_architecture.png)
在上图中，{S0，S1，...，SN-1}是图像描述中所用到的单词（中文环境下使用jieba分词），{wes0，wes1，...，wesN-1}是它们相应的单词嵌入向量。LSTM的输出{p1，p2，...，pN}是以当前词和上一层的LSTM输出作为输入，生成的下一个单词概率分布。{log p1(s1)、log p2(s2)、...、log pN(sN)}是每一步中正确单词的对数似然；这些对数似然的相反数之和是模型的最小化目标。

### 训练过程

在训练的第一阶段，Inception v3模型的参数是固定的，它只是一个静态图像编码函数。在Inception v3模型的末端添加一个可训练层，以将图像嵌入向量转换为单词嵌入向量空间。模型要训练的就是单词嵌入的参数、inception v3可训练层的参数以及LSTM的参数。在训练的第二阶段，训练所有参数（包括inception v3的参数）对整个编码和解码的模型进行微调。


### 描述生成过程

给定一个训练好的模型和图像，我们使用波束搜索来生成该图像的自然语言描述。描述是逐字生成的，其中在每个步骤t中，我们使用已经生成的长度为t-1的句子集来生成长度为t的新句子集。在每个步骤中，我们只保留前k个候选项，其中超参数k称为波束大小。我们发现当k＝3时嫩获得最佳性能。

## 运行

### 数据集处理
在AI Challenger 注册后，可[在此](https://challenger.ai/dataset/caption)下载数据集。数据集由两部分构成：包含图片的目录和一个包含描述的json文件。每张图片对应五个句子描述。  
**build_AIChallenge_data.py**将数据集解析成tensorflow标准的TFRecord文件。运行之前需要修改以下两处：
  
* 在代码头部修改待处理数据集和输出的文件路径 *train\_image\_dir*, *train\_caption\_dir*, *output\_dir*
* 在*main*中配置如何分配训练集，验证集和测试集的大小

```
	train_cutoff = int(0.8 * len(AI_Challenger_val_dataset))
    val_cutoff = int(0.90 * len(AI_Challenger_val_dataset))

    train_dataset = AI_Challenger_val_dataset[:train_cutoff]
    val_dataset = AI_Challenger_val_dataset[train_cutoff:val_cutoff]
    test_dataset = AI_Challenger_val_dataset[val_cutoff:]

```
为了生成image-caption对，这里相当于将每张图片复制了五次。因此输出文件将会很大。
由于磁盘空间和CPU算力限制，本人只采用了AI Challenger测试集中的30000张图片，进行了8:1:1的分配，即训练集24000张，验证集与测试集各3000张图片。


### 训练与验证
首先下载inception_v3 模型检查点文件： [下载地址](https://github.com/tensorflow/models/tree/master/research/slim#tensorflow-slim-image-classification-library)  
**train.py**  运行之前修改模型输入、inception_v3检查点路径和模型检查点输出路径:   
*input\_file\_pattern*，*inception\_checkpoint\_file*, *train\_dir*  
**evaluate.py**  运行之前修改验证数据输入，模型检查点输入以及验证事件存储的路径
*input\_file\_pattern*，*checkpoint|_dir*, *eval\_dir*  
训练和验证开启两个python进程同时进行。  
开启第三个进程运行tensorboard 可视化训练过程  

```
            tensorboard --logdir="path/to/model/dir"                                           
```



### 生成图像描述
**run_inference.py**  运行之前修改模型检查点，字典文件（数据集处理时生成）和要处理的图片路径  
*checkpoint_path*, *vocab_file*, *input_files*
当训练生成第一个检查点后即可运行进行测试。

