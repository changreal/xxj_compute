# 小学鸡计算baseline说明

（起这个名字不知道能不能防窥）

## 依赖
实测环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.8.8

总体是基于bert4keras的`task_seq2seq_ape210k_math_word_problem.py`改的

## 数据集
放在`datasets`文件夹下面

其中`*.ape.json`格式的文件是ape210k数据集

其中`test.csv`（在baseline里用作赛道数据集的测试集）, `train.csv`是赛道数据训练集

## 预训练模型权重加载

baseline使用了`BERT-wwm-ext`和`roberta-wwm-ext`两个中文预训练模型，都是base大小，保存在`weights`文件夹下面，
使用的时候记得把模型权重文件放到对应目录文件里。（详细见代码加载权重部分就知道怎么放了）

中文预训练语言模型bert系列权重下载：https://github.com/ymcui/Chinese-BERT-wwm

bert4keras调用权重下载见：https://github.com/bojone/bert4keras


## 训练与评估


用bert训练
```shell
sh run_bert.sh --do_train
```

用bert评估
```
sh run_bert.sh 
```

用roberta训练
```
sh run_roberta.sh --do_train 
```

用roberta评估
```
sh run_roberta.sh 
```

PS: 运行不了脚本的话改一下shell文件的编码，就是
```
vi run_bert.sh
:set ff=unix
:wq
```

具体超参数见配置shell文件，运行的时候调整参数修改shell文件运行
默认超参数是：
> batch_size=16 (原来是32，服务器gpu限制改成了16)   
maxlen=160      
epoches=100   
learning_rate=2e-5


## fine-tune后的模型权重加载
默认调用在ape验证集上效果最好的权重，在分别是根目录底下的`bert_bert_model.weights`和`best_reberta_model.weights`

## 输出
模型运行中会把符合赛道要求格式的`.csv`文件保存到`outputs`文件夹里，把`.csv`文件提交赛道系统就可以进行分数评估了

模型运行的时候保存的最佳的权重文件在根目录下的`bert_bert_model.weights`或者`bert_robert_model.weights`文件上。

详细输出命名与保存位置见代码


## 待改进
- 让模型支持断点续训，规范断点权重与输出文件命名
- 支持多gpu训练
- 从结构上、引入启发规则上、训练方式上改进baseline
- 调参
- 拥抱赛道数据集
