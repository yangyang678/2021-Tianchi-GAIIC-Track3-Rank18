# 2021-Tianchi-GAIIC-Track3-Rank18
天池2021"全球人工智能技术创新大赛"【赛道三】：小布助手对话短文本语义匹配 - 第18名解决方案

赛题链接：https://tianchi.aliyun.com/competition/entrance/531851/rankingList

**1、任务背景**

该赛题出题方是OPPO公司研发的语音助手-小布助手，它们的主要业务是对话式服务。意图识别是对话系统中的一个核心任务，而对话短文本语义匹配是意图识别的主流算法方案之一。

本赛题要求参赛队伍根据脱敏后的短文本query-pair，预测它们是否属于同一语义。

初赛训练样本10万，复赛训练样本30万。初赛测试集2.5万条，复赛测试集5万条。

**2、评价指标**

初赛指标为AUC，复赛阶段同时考虑性能标准和AUC，性能标准是约束条件，在复赛阶段需要在限定时间内完成预测。

为了合理分配资源，天池平台上**单次提交运行时间**不能超过**80**个小时，超出后程序自动停止。

复赛会限定5万条测试集采用逐条预测的方式，只能使用CPU算力和一张GPU卡算力，总预测时间不超过15分钟（含网络调用开销），超时默认未预测样本预测结果为0，换算下来就是预测每条测试样本不超过18ms。

**3、使用方案**

从预训练、微调、推理三个方面入手，完整设计并实现了一个应用于脱敏句子对关系预测的方案：

 1.在预训练层面，借鉴**MLM**自监督模型的思想，应用**动态MASK、N-gram MASK、数据增强**等策略进行预训练任务 

2.对脱敏语料的密文字表与开源通用语料的明文字表按照字频进行对应，从而完成脱敏语料的**字向量初始化**，提升模型的收敛速度 

3.在微调阶段，尝试对抗训练、warmup、权重衰减等策略，使用**分块shuffle、混合精度训练、分布式训练**等手段提升模型的训练速度 

 4.在推理任务上，由于比赛对推理时间做了限制，更符合工业界场景。因此尝试使用**ONNX-Runtime**进行推理加速

**4、模型细节**

比赛总共使用了三个模型，包括bert、roberta以及nezha。

**5、数据**

本项目没有提供数据，如果需要数据，请到天池比赛主页下载

**6、预训练模型准备**

模型下载：

nezha-cn-wwm：

https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/NEZHA-TensorFlow

nezha-cn-base：

https://huggingface.co/miaomiaomiao/nezha_miao

bert-cn-wwm: 

https://github.com/ymcui/Chinese-BERT-wwm

roberta-cn-uer: 

https://huggingface.co/uer/chinese_roberta_L-12_H-768

**7、实验环境**

1.PyTorch	1.5.0

2.cudnn	7

3.cuda	10.2

4.onnxruntime-gpu=1.4

5.transformers 4.3.3

**8、端到端训练脚本**
```
sh run.sh
```

**9、trick细节**

详见blog: http://119.91.118.115/competition/36.html
