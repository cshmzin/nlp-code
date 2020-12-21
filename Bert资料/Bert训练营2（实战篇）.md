```python
# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
!ls /home/aistudio/data
```


```python
# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
!ls /home/aistudio/work
```


```python
# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
!mkdir /home/aistudio/external-libraries
!pip install transformers==3.4.0 # 直接执行此步安转
# !pip install beautifulsoup4 -t /home/aistudio/external-libraries
```


```python
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')
```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

# 1 BERT的token细节

## 1.1 CLS与SEP

<img src="https://ai-studio-static-online.cdn.bcebos.com/cc56aba3ec5a408884081042a364372d04f6a3e0b48e4e04b4299a7c7b63df1a" width="600" />

上图是BERT模型输入Embedding的过程，注意到两个特殊符号，一个是[CLS]，一个是[SEP]。在序列的开头添加的[CLS]主要是用来学习整个句子或句子对之间的语义表示。[SEP]主要是用来分割不同句子。

之所以会选择[CLS]，因为与文本中已有的其他词相比，这个无明显语义信息的符号会更公平地融合文本中各个词的语义信息，从而更好的表示整句话的语义。

## 1.2 对应token位置的输出

有了各种各样的token输入之后，BERT模型的输出是什么呢。通过下图能够看出会有两种输出，一个对应的是红色框，也就是对应的[CLS]的输出，输出的shape是[batch size，hidden size]；另外一个对应的是蓝色框，是所有输入的token对应的输出，它的shape是[batch size，seq length，hidden size]，这其中不仅仅有[CLS]对于的输出，还有其他所有token对应的输出。

<img src="https://ai-studio-static-online.cdn.bcebos.com/4086fdccdbf84cd0b2638a5b1f0ab8dbd1c3270a142445618c552dbfa5bed9dd" width="500" />

在使用代码上就要考虑到底是使用第一种输出，还是第二种了。大部分情况是是会选择[CLS]的输出，再进行微调的操作。不过有的时候使用所有token的输出也会有一些意想不到的效果。

BertPooler就是代表的就是[CLS]的输出，可以直接调用。大家可以修改下代码，使其跑通看看。


```python
import torch
from torch import nn
```


```python
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        # hidden_states.shape 为[batch_size, seq_len, hidden_dim]
        # assert hidden_states.shape == torch.Size([8, 768])
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```


```python
class Config:
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.attention_probs_dropout_prob = 0.1

config = Config()
bertPooler = BertPooler(config)
input_tensor = torch.ones([8, 50, 768])
output_tensor = bertPooler(input_tensor)

assert output_tensor.shape == torch.Size([8, 50, 768])
```

上面的代码会报错吧，看看错在哪里，有助于大家理解输出层的维度。

## 1.3 BERT的Tokenizer

<img src="https://ai-studio-static-online.cdn.bcebos.com/cc56aba3ec5a408884081042a364372d04f6a3e0b48e4e04b4299a7c7b63df1a" width="600" />

我们再看看上面这张关于BERT模型的输入的图，我们会发现，在input这行，对于英文的输入是会以一种subword的形式进行的，比如playing这个词，是分成play和##ing两个subword。那对于中文来说，是会分成一个字一个字的形式。这么分subword的好处是减小了字典vocab的大小，同时会减少OOV的出现。那像playing那样的分词方式是怎么做到呢，subword的方式非常多，BERT采用的是wordpiece的方法，具体知识可以阅读补充资料[《深入理解NLP Subword算法：BPE、WordPiece、ULM》](https://zhuanlan.zhihu.com/p/86965595)。

BERT模型预训练阶段的vocab，可以点击data/data56340/vocab.txt查看。

下图截了一部分，其中[unused]是可以自己添加token的预留位置，101-104会放一些特殊的符号，这样大家就明白第一节最后代码里添加102的含义了吧。

![](https://ai-studio-static-online.cdn.bcebos.com/73cf297f80254c0abd9348160c4ccc64c0e954ea4023426694e043feb9cb6715)

在实际代码过程中，有关tokenizer的操作可以见Transformers库中tokenization_bert.py。

里面有很多的可以操作的接口，大家可以自行尝试，下面列了其中一个。


```python
from typing import List, Optional, Tuple
def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
    """
    Build model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens.
    A BERT sequence has the following format:

    - single sequence: ``[CLS] X [SEP]``
    - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    Args:
        token_ids_0 (:obj:`List[int]`):
            List of IDs to which the special tokens will be added.
        token_ids_1 (:obj:`List[int]`, `optional`):
            Optional second list of IDs for sequence pairs.

    Returns:
        :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
    """
    if token_ids_1 is None:
        return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
    cls = [self.cls_token_id]
    sep = [self.sep_token_id]
    return cls + token_ids_0 + sep + token_ids_1 + sep
```

大家改改下面的code试一试。


```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('/home/aistudio/data/data56340')
inputs_1 = tokenizer("欢迎大家来到后厂理工学院学习。")
print(inputs_1)

inputs_2 = tokenizer("欢迎大家来到后厂理工学院学习。", "hello")
print(inputs_2)

inputs_3 = tokenizer.encode("欢迎大家来到后厂理工学院学习。", "hello")
print(inputs_3)

inputs_4 = tokenizer.build_inputs_with_special_tokens(inputs_3)
print(inputs_4)
```

# 2 MLM和NSP预训练任务

此阶段我们开始对两个BERT的预训练任务展开学习，Let`s go！

## 2.1 MLM

如何理解MLM，可以先从LM（language model，语言模型）入手，LM的目地是基于上文的内容来预测下文的可能出现的词，由于LM是单向的，要不从左到右要不从右到左，很难做到结合上下文语义。为了改进LM，实现双向的学习，MLM就是一种，通过对输入文本序列随机的mask，然后通过上下文来预测这个mask应该是什么词，至此解决了双向的问题。这个任务的表现形式更像是完形填空，通过此方向使得BERT完成自监督的学习任务。

<img src="https://ai-studio-static-online.cdn.bcebos.com/7afc59b371a14e499f0b5d9fa205a5089381f5111a9f468eb986966e4e3087f9" width="450" />

那随机的mask是怎么做的呢？具体的做法是，将每个输入的数据句子中15%的概率随机抽取token，在这15%中的80%概论将token替换成[MASK]，如上图所示，15%中的另外10%替换成其他token，比如把‘理’换成‘后’，15%中的最后10%保持不变，就是还是‘理’这个token。

之所以采用三种不同的方式做mask，是因为后面的fine-tuning阶段并不会做mask的操作，为了减少pre-training和fine-tuning阶段输入分布不一致的问题，所以采用了这种策略。

如果使用MLM，它的输出层可以参照下面代码，取自Transformers库中modeling_bert.py。


```python
class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 这部操作加了一些全连接层和layer归一化
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is an output-only bias for each token.
        # 在nn.Linear操作过程中的权重和bert输入的embedding权重共享，思考下为什么需要共享？原因见下面描述。
        # self.decoder在预测生成token的概论
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # decoder层虽然权重是共享的，但是会多一个bias偏置项，在此设置
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
```

Embedding层和FC层（上面代码nn.Linear层）权重共享。

Embedding层可以说是通过onehot去取到对应的embedding向量，FC层可以说是相反的，通过向量（定义为 v）去得到它可能是某个词的softmax概率，取概率最大（贪婪情况下）的作为预测值。那哪一个会是概率最大的呢？Embedding层和FC层权重共享，Embedding层中和向量 v 最接近的那一行对应的词，会获得更大的预测概率。实际上，Embedding层和FC层有点像互为逆过程。

通过这样的权重共享可以减少参数的数量，加快收敛。

我们有了BertLMPredictionHead后，就可以完成MLM的预训练任务了。有两种选择，第一个是BertOnlyMLMHead，它是只考虑单独MLM任务的，通过BertForMaskedLM完成最终的预训练，Loss是CrossEntropyLoss；第二个是BertPreTrainingHeads，它是同时考虑MLM和NSP任务的，通过BertForPreTraining完成，Loss是CrossEntropyLoss。原本论文肯定是第二种MLM和NSP一块训练的，但如果有单独训练任务需求是使用者可自行选择。

以上提到的如BertOnlyMLMHead类，可以查阅Transformers库modeling_bert.py。

## 2.2 NSP

BERT的作者在设计任务时，还考虑了两个句子之间的关系，来补充MLM任务能力，设计了Next Sentence Prediction（NSP）任务，这个任务比较简单，NSP取[CLS]的最终输出进行二分类，来判断输入的两个句子是不是前后相连的关系。

构建数据的方法是，对于句子1，句子2以50%的概率为句子1相连的下一句，以50%的概率在语料库里随机抽取一句。以此构建了一半正样本一半负样本。

<img src="https://ai-studio-static-online.cdn.bcebos.com/7a7517960022432697937c9b6e1411bf4d58c435301642c78cd220a411b55615" width="450" />

从上图可以看出，NSP任务实现比较简单，直接拿[CLS]的输出加上一个全连接层实现二分类就可以了。

```
self.seq_relationship = nn.Linear(config.hidden_size, 2)
```

最后采用CrossEntropyLoss计算损失。

# 3 代码实操预训练

BERT预训任务分为MLM和NSP，后续一些预训练模型的尝试发现，NSP任务其实应该比较小，所以如果大家在预训练模型的基础上继续训练，可以直接跑MLM任务。

## 3.1 mask token 处理

在进行BERT的预训练时，模型送进模型的之前需要对数据进行mask操作，处理代码如下：


```python
def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    # 调出[MASK]
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens  

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels
```

## 3.2 大型模型训练策略

对于BERT的预训练操作，会涉及很多训练策略，目地都是解决如何在大规模训练时减少训练时间，充分利用算力资源。以下代码实例。


```python
# gradient_accumulation梯度累加
# 一般在单卡GPU训练时常用策略，以防止显存溢出
if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
```


```python
# Nvidia提供了一个混合精度工具apex
# 实现混合精度训练加速
if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
```


```python
# multi-gpu training (should be after apex fp16 initialization)
# 一机多卡
if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)
```


```python
# Distributed training (should be after apex fp16 initialization)
# 多机多卡分布式训练
if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )
```

以上代码都是常添加在BERT训练代码中的策略方法，这里提供一个补充资料[《神经网络分布式训练、混合精度训练、梯度累加...一文带你优雅地训练大型模型》](https://zhuanlan.zhihu.com/p/110278004)。

在训练策略上，基于Transformer结构的大规模预训练模型预训练和微调都会采用wramup的方式。
```
scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
```
那BERT中的warmup有什么作用呢？

在预训练模型训练的开始阶段，BERT模型对数据的初始分布理解很少，在第一轮训练的时候，模型的权重会迅速改变。如果一开始学习率很大，非常有可能对数据产生过拟合的学习，后面需要很多轮的训练才能弥补，会花费更多的训练时间。但模型训练一段时间后，模型对数据分布已经有了一定的学习，这时就可以提升学习率，能够使得模型更快的收敛，训练也更加稳定，这个过程就是warmup，学习率是从低逐渐增高的过程。

那为什么warmup之后会有decay的操作？

当BERT模型训练一定时间后，尤其是后续快要收敛的时候，如果还是比较大的学习率，比较难以收敛，调低学习率能够更好的微调。

更多的思考可以阅读[《神经网络中 warmup 策略为什么有效；有什么理论解释么？》](https://www.zhihu.com/question/338066667/answer/771252708)。

好了，预训练的知识基本就这些了，挖的比较深。

如果你想自己来一些预训练的尝试，可以github上找一份源码，再去找一个中文数据集试一试。

如果只是想用一用BERT，那就可以继续下一节课微调模型的学习，以后的工作中大部分时间会花在处理微调模型的过程中。

同学们加油！

# 4 BERT微调细节详解

上面我们已经对BERT的预训练任务有了深刻的理解，本环节将对BERT的Fine-tuning微调展开探讨。

预训练+微调技术掌握熟练后，就可以在自己的业务上大展身教了，可以做一些大胆的尝试。

## 4.1 BERT微调任务介绍

微调（Fine-tuning）是在BERT强大的预训练后完成NLP下游任务的步骤，这也是所谓的迁移策略，充分应用大规模的预训练模型的优势，只在下游任务上再进行一些微调训练，就可以达到非常不错的效果。

下图是BERT原文中微调阶段4各种类型的下游任务。其中包括：
* 句子对匹配（sentence pair classification）
* 文本分类（single sentence classification）
* 抽取式问答（question answering）
* 序列标注（single sentence tagging）

![](https://ai-studio-static-online.cdn.bcebos.com/6b73bf80f9c44975a5a0a0a12647372c18bbebb0f7bb4b198777861b80cdbb57)

## 4.2 文本分类任务

我们先看看文本分类任务的基本微调操作。如下图所示，最基本的做法就是将预训练的BERT读取进来，同时在[CLS]的输出基础上加上一个全连接层，全连接层的输出维度就是分类的类别数。

![](https://ai-studio-static-online.cdn.bcebos.com/b31f31684d2b4d74af5df9f41ffe467d6bbc1b0b67344aea88545472d8b06643)


从代码实现上看可以从两个角度出发：

1.直接调用Transformers库中BertForSequenceClassification类实现，代码如下：


```python
import torch
import torch.nn as nn

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 考虑多分类的问题
        self.num_labels = config.num_labels
        # 调用bert预训练模型
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 在预训练的BERT上加上一个全连接层，用于微调分类模型
        # config.num_labels是分类数
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
```

2.如果想做一些更复杂的微调模型，可以参照上述封装好的类，写一个自己需要的微调层满足分类的需求，代码如下：


```python
class NewModel(nn.Module):
    def __init__(self):
        super(NewModel, self).__init__()
        # 调用bert预训练模型
        self.model = BertModel.from_pretrained(modelPath)  
        # 可以自定义一些其他网络做为微调层的结构
        self.cnn = nn.Conv2d()
        self.rnn = nn.GRU()
        self.dropout = nn.Dropout(0.1)
        # 最后的全连接层，用于分类
        self.l1 = nn.Linear(768, 2)
```

对比一下上述两个类，你会发现如果是调用Transformers中的BertForSequenceClassification，加载bert预训练模型仅传了一个config，而自己创建类，要传整个预训练模型的路径（其中包括config和model文件）。大家思考下，看看源码寻找答案？

## 4.3 文本匹配任务


接着我们看下匹配问题是如何搭建的，网络结构如下图所示。

![](https://ai-studio-static-online.cdn.bcebos.com/e87fa90d48354aa98bb4dfb78d6fa36a23e977beffa24ff49ee7aa0c3c6abc29)

虽然文本匹配问题的微调结构和分类问题有一定的区别，它的输入是两个句子，但是它最终的输出依然是要做一个二分类的问题，所以如果你想用BERT微调一个文本匹配模型，可以和分类问题用的代码是一样的，依然可以采用Transformers库中BertForSequenceClassification类实现，只不过最终全连接层输出的维度为2。

tips：实际在工程中，经过大量的验证，如果直接采用上述的BERT模型微调文本匹配问题，效果不一定很好。一般解决文本匹配问题会采用一些类似孪生网络的结构去解决，该课就不过多介绍了。

## 4.4 序列标注任务

下面我们看一下序列标注问题，BERT模型是如何进行微调的。下图是原论文中给出的微调结构图。


![](https://ai-studio-static-online.cdn.bcebos.com/5f913ebbb5104bf6a7419ab12fa88c90d778c073baa54136858c824f006c16f5)

理解序列标注问题，要搞清楚它主要是在做什么事情。一般的分词任务、词性标注和命名体识别任务都属于序列标注问题。这类问题因为输入句子的每一个token都需要预测它们的标签，所以序列标注是一个单句多label分类任务，BERT模型的所有输出（除去特殊符号）都要给出一个预测结果。

同时，我们要保证BERT的微调层的输出是[batch_size, seq_len, num_labels]。

如果继续使用Transformers库，可以直接调用BertForTokenClassification类。部分代码如下：


```python
class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 序列标注的类别数
        self.num_labels = config.num_labels
        # 调用BERT预训练模型，同时关掉pooling_layer的输出，原因在上段有解释。
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 增加一个微调阶段的分类器，对每一个token都进行分类
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
```

同理，如果想进一步提升序列标注的性能，也是要自己增加一些层，感兴趣的可以自己试试啊。

## 4.5 问答任务

论文里还有最后一种微调结构，就是抽取式的QA微调模型，该问题是在[SQuAD1.1](https://rajpurkar.github.io/SQuAD-explorer/)设计的，如下图所示。

![](https://ai-studio-static-online.cdn.bcebos.com/2d4ec0b2c0554b3a88c0f78913facbe11251145bad1a406eb39b833c497f1958)


QA问题的微调模型搭建也不难，一些初始化的操作见下面代码（源自Transformers库）：



```python
class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 判断token是答案的起点和终点的类别，也就是一个二分类的问题，此处应该等于2
        self.num_labels = config.num_labels
        # 导入BERT的预训练模型，同时不输出pooling层，那就是把所有token对应的输出都保留
        # 输出维度是[batch_size, seq_len, embedding_dim]
        self.bert = BertModel(config, add_pooling_layer=False)
        # 通过一个全连接层实现抽取分类任务
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
```

说到这里，大家可能还是不太好理解QA问题的微调过程，我们在看下相对应的forward代码。


```python
def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    start_positions=None,
    end_positions=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    # 拿到所有token的输出
    sequence_output = outputs[0]
    # 得到每个token对应的分类结果，就是分为start位置和end位置的概论
    logits = self.qa_outputs(sequence_output)
    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    total_loss = None
    if start_positions is not None and end_positions is not None:
        # If we are on multi-GPU, split add a dimension
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)
        # sometimes the start/end positions are outside our model inputs, we ignore these terms
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)
        # 通过交叉熵来计算loss
        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2

    if not return_dict:
        output = (start_logits, end_logits) + outputs[2:]
        return ((total_loss,) + output) if total_loss is not None else output
    
    # 结果是要返回start和end的结果
    return QuestionAnsweringModelOutput(
        loss=total_loss,
        start_logits=start_logits,
        end_logits=end_logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
```

以上四个任务就是BERT原论文中提到的微调任务，实现方式大体都比较相像，在实际的使用过程中可以借鉴。

# 5 微调模型的设计问题

## 5.1 预训练模型输入长度的限制

我们通过对BERT预训练模型的了解，可以知道，BERT预设的最大文本长度为512。
```
# Transformers源码configuration_bert.py中的定义
def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512, # 通过这个参数可以得知预训练bert的长度
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        **kwargs
    ):
```
也就是说，BERT模型要求输入句子的长度不能超过512，同时还要考虑[CLS]这些特殊符号的存在，实际文本的长度会更短。

究其原因，随着文本长度的不断增加，计算所需要的显存也会成线性增加，运行时间也会随着增长。所以输入文本的长度是需要加以控制的。

在实际的任务中我们的输入文本一般会有两个方面，要不就是特别长，比如文本摘要、阅读理解任务，它们的输入文本是有可能超过512；另外一种就是一些短文本任务，如短文本分类任务。

下面我们会给出一些方法。

## 5.2 长文本问题

说到长文本处理，最直接的方法就是截断。

由于 Bert 支持最大长度为 512 个token，那么如何截取文本也成为一个很关键的问题。

[《How to Fine-Tune BERT for Text Classification?》](https://arxiv.org/abs/1905.05583)给出了几种解决方法:

* head-only： 保存前 510 个 token （留两个位置给 [CLS] 和 [SEP] ）
* tail-only： 保存最后 510 个token
* head + tail ： 选择前128个 token 和最后382个 token

作者是在IMDB和Sogou News数据集上做的试验，发现head+tail效果会更好一些。但是在实际的问题中，大家还是要人工的筛选一些数据观察数据的分布情况，视情况选择哪种截断的方法。

除了上述截断的方法之外，还可以采用sliding window的方式做。

用划窗的方式对长文本切片，分别放到BERT里，得到相对应的CLS，然后对CLS进行融合，融合的方式也比较多，可以参考以下方式：

* max pooling最大池化
* avg pooling平均池化
* attention注意力融合
* transformer等

相关思考可以参考：[《Multi-passage BERT: A Globally Normalized BERT Model for
Open-domain Question Answering》](https://arxiv.org/pdf/1908.08167.pdf)和[《PARADE: Passage Representation Aggregation for Document Reranking》](https://arxiv.org/pdf/2008.09093.pdf)

## 5.3 短文本问题

在遇到一些短文本的NLP任务时，我们可以对输入文本进行一定的截断，因为过长的文本会增加相应的计算量。

那如何选取短文本的输入长度呢？需要大家对数据进行简单的分析。虽然简单，但这往往是工作中必须要注意的细节。

## 5.4 微调层的设计

针对不同的任务大家可以继续在bert的预训练模型基础上加一些网络的设计，比如文本分类上加一些cnn；比如在序列标注上加一些crf等等。

往往可以根据经验进行尝试。

### 5.4.1 Bert+CNN

CNN结构在学习一些短距离文本特征上有一定的优势，可以和Bert进行结合，会有不错的效果。

下图是TextCNN算法的结构示意图，同学们可以尝试补全下面代码，完成Bert和TextCNN的结合。

<img src="https://ai-studio-static-online.cdn.bcebos.com/3cf5e32f41b94137903f288dda62879177d0d4d9b37049abb14e05e36ca514ee" width="800" />


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertModel


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, filter_sizes):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.init_params()

    def init_params(self):
        for m in self.convs:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)

    def forward(self, x):
        return [F.relu(conv(x)) for conv in self.convs]
```


```python
class BertCNN(BertPreTrainedModel):

    def __init__(self, config, num_labels, n_filters, filter_sizes):
        # total_filter_sizes = "2 2 3 3 4 4"
        # filter_sizes = [int(val) for val in total_filter_sizes.split()]
        # n_filters = 6
        super(BertCNN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.convs = Conv1d(config.hidden_size, n_filters, filter_sizes)

        self.classifier = nn.Linear(len(filter_sizes) * n_filters, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
            input_ids: 词对应的 id
            token_type_ids: 区分句子，0 为第一句，1表示第二句
            attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
        """
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # encoded_layers: [batch_size, seq_len, bert_dim=768]
        
        encoded_layers = self.dropout(encoded_layers)
        """
        one code # 对encoded_layers做维度调整

        one code # 调用conv层

        one code # 图中所示采用最大池化融合
        """
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat: [batch_size, filter_num * len(filter_sizes)]

        logits = self.classifier(cat)
        # logits: [batch_size, output_dim]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

```

上面代码共有三行需要填写，主要是TextCNN结构的逻辑，大家要多加思考。

填完后，可以参照下面代码答案。

```
class BertCNN(nn.Module):

    def __init__(self, config, num_labels, n_filters, filter_sizes):
        super(BertCNN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.convs = Conv1d(config.hidden_size, n_filters, filter_sizes)

        self.classifier = nn.Linear(len(filter_sizes) * n_filters, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
            input_ids: 词对应的 id
            token_type_ids: 区分句子，0 为第一句，1表示第二句
            attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
        """
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        # encoded_layers: [batch_size, seq_len, bert_dim=768]
        
        encoded_layers = self.dropout(encoded_layers)
        """
        one code # 对encoded_layers做维度调整

        one code # 调用conv层

        one code # 图中所示采用最大池化融合
        """
        encoded_layers = encoded_layers.permute(0, 2, 1)
        # encoded_layers: [batch_size, bert_dim=768, seq_len]

        conved = self.convs(encoded_layers)
        # conved 是一个列表， conved[0]: [batch_size, filter_num, *]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        # pooled 是一个列表， pooled[0]: [batch_size, filter_num]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat: [batch_size, filter_num * len(filter_sizes)]

        logits = self.classifier(cat)
        # logits: [batch_size, output_dim]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
```

### 5.4.2 Bert+LSTM

那要是想加上一个lstm呢？参照下面代码。


```python
class BertLSTM(BertPreTrainedModel):

    def __init__(self, config, num_labels, rnn_hidden_size, num_layers, bidirectional, dropout):
        super(BertLSTM, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.rnn = nn.LSTM(config.hidden_size, rnn_hidden_size, num_layers,bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        encoded_layers = self.dropout(encoded_layers)
        # encoded_layers: [batch_size, seq_len, bert_dim]

        _, (hidden, cell) = self.rnn(encoded_layers)
        # outputs: [batch_size, seq_len, rnn_hidden_size * 2]
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 连接最后一层的双向输出

        logits = self.classifier(hidden)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
```

### 5.4.3 Bert+attention

当然，你也可以加一个attention。


```python
class BertATT(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    """

    def __init__(self, config, num_labels):
        super(BertATT, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.W_w = nn.Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.u_w = nn.Parameter(torch.Tensor(config.hidden_size, 1))

        nn.init.uniform_(self.W_w, -0.1, 0.1)
        nn.init.uniform_(self.u_w, -0.1, 0.1)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
            input_ids: 词对应的 id
            token_type_ids: 区分句子，0 为第一句，1表示第二句
            attention_mask: 区分 padding 与 token， 1表示是token，0 为padding
        """
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        encoded_layers = self.dropout(encoded_layers)
        # encoded_layers: [batch_size, seq_len, bert_dim=768]

        score = torch.tanh(torch.matmul(encoded_layers, self.W_w))
        # score: [batch_size, seq_len, bert_dim]

        attention_weights = F.softmax(torch.matmul(score, self.u_w), dim=1)
        # attention_weights: [batch_size, seq_len, 1]

        scored_x = encoded_layers * attention_weights
        # scored_x : [batch_size, seq_len, bert_dim]

        feat = torch.sum(scored_x, dim=1)
        # feat: [batch_size, bert_dim=768]
        logits = self.classifier(feat)
        # logits: [batch_size, output_dim]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
```

# 6 微调阶段的调整策略

## 6.1不同学习率的设置

在[《How to Fine-Tune BERT for Text Classification?》](https://arxiv.org/abs/1905.05583)一文中作者提到了一个策略。

这个策略叫作 slanted triangular（继承自 ULM-Fit）。它和 BERT 的原版方案类似，都是带 warmup 的先增后减。通常来说，这类方案对初始学习率的设置并不敏感。但是，在 fine-tune阶段使用过大的学习率，会打乱 pretrain 阶段学习到的句子信息，造成“灾难性遗忘”。

比如下方的图（源于论文），最右边学习率=4e-4的loss已经完全无法收敛了，而学习率=1e-4的loss曲线明显不如学习率=2e-5和学习率=5e-5的低。

综上所述，对于BERT模型的训练和微调学习率取2e-5和5e-5效果会好一些。

![](https://ai-studio-static-online.cdn.bcebos.com/33c00797a66b417b9c0b4e2b6c9456534a1867d7e0f14bb0a4f10b5cadd37817)

不过对于上述的学习率针对的是BERT没有下游微调结构的，是直接用BERT去fine-tune。

那如果微调的时候接了更多的结构，是不是需要再考虑下学习率的问题呢？大家思考一下？


答案是肯定的，我们需要考虑不同的学习率来解决不同结构的问题。比如BERT+TextCNN，BERT+BiLSTM+CRF，在这种情况下。

BERT的fine-tune学习率可以设置为5e-5, 3e-5, 2e-5。

而下游任务结构的学习率可以设置为1e-4，让其比bert的学习更快一些。

至于这么做的原因也很简单：BERT本体是已经预训练过的，即本身就带有权重，所以用小的学习率很容易fine-tune到最优点，而下接结构是从零开始训练，用小的学习率训练不仅学习慢，而且也很难与BERT本体训练同步。

为此，我们将下游任务网络结构的学习率调大，争取使两者在训练结束的时候同步：当BERT训练充分时，下游任务结构也能够训练充分。

## 6.2 weight decay权重衰减

权重衰减等价于L2范数正则化。正则化通过为模型损失函数添加惩罚项使得学习的模型参数值较小，是常用的过拟合的常用手段。

权重衰减并不是所有的权重参数都需要衰减，比如bias，和LayerNorm.weight就不需要衰减。

具体实现可以参照下面部分代码。


```python
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# 对应optimizer_grouped_parameters中的第一个dict，这里面的参数需要权重衰减
need_decay = []
for n, p in model.named_parameters():
    if not any(nd in n for nd in no_decay):
        need_decay.append(p)
        
# 对应optimizer_grouped_parameters中的第二个dict，这里面的参数不需要权重衰减
not_decay = []
for n, p in model.named_parameters():
    if any(nd in n for nd in no_decay):
        not_decay.append(p)
        
# AdamW是实现了权重衰减的优化器
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
criterion = nn.CrossEntropyLoss()
```

## 6.3 实战中的迁移策略


那拿到一个BERT预训练模型后，我们会有两种选择：

1. 把BERT当做特征提取器或者句向量，不在下游任务中微调。
2. 把BERT做为下游业务的主要模型，在下游任务中微调。

具体的使用策略要多加尝试，没有绝对的正确。

那如何在代码中控制BERT是否参与微调呢？代码如下：


```python
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        init_checkpoint = config['init_checkpoint']
        freeze_bert = config['freeze_bert']
        dropout = config['dropout']
        self.use_bigru = config['use_bigru']
        self.output_hidden_states = config['output_hidden_states']
        self.concat_output = config['concat_output']

        self.config = config

        bert_config = BertConfig.from_pretrained(os.path.join(init_checkpoint, 'bert_config.json'),
                                                 output_hidden_states=self.output_hidden_states)
        self.model = BertModel.from_pretrained(os.path.join(init_checkpoint, 'pytorch_model.bin'),
                                               config=bert_config)
        self.dropout = nn.Dropout(dropout)
        
        # bert是否参与微调，可以通过一下代码实现
        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False  # 亦可以针对性的微调或者冻结某层参数

        if self.use_bigru:
            self.biGRU = torch.nn.GRU(768, 768, num_layers=1, batch_first=True, bidirectional=True)
            self.dense = nn.Linear(bert_config.hidden_size * 2, 3)  # 连接bigru的输出层
        elif self.concat_output:
            self.dense = nn.Linear(bert_config.hidden_size * 3, 3)  # 连接concat后的三个向量
        else:
            self.dense = nn.Linear(bert_config.hidden_size, 3)  # 输出3维（3分类）

```

那如果有选择的进行bert某些层的冻结可以参照以下代码。


```python
# Freeze parts of pretrained model
# config['freeze'] can be "all" to freeze all layers,
# or any number of prefixes, e.g. ['embeddings', 'encoder']
if 'freeze' in config and config['freeze']:
    for name, param in self.base_model.named_parameters():
        if config['freeze'] == 'all' or 'all' in config['freeze'] or name.startswith(tuple(config['freeze'])):
            param.requires_grad = False
            logging.info(f"Froze layer {name}...")
```


```python
if freeze_embeddings:
    for param in list(model.bert.embeddings.parameters()):
        param.requires_grad = False
        print ("Froze Embedding Layer")

# freeze_layers is a string "1,2,3" representing layer number
if freeze_layers is not "":
    layer_indexes = [int(x) for x in freeze_layers.split(",")]
    for layer_idx in layer_indexes:
            for param in list(model.bert.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False
            print ("Froze Layer: ", layer_idx)
```

# 7 完成你的BERT任务（作业在其中）

在做项目前可以执行下列语句安装所需的库。


```python
# 也可以在终端里安装，注意下版本
!pip install transformers==3.4.0
```

该部分项目采用数据集为[中文文本分类数据集THUCNews](http://thuctc.thunlp.org/)。

THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。我们在原始新浪新闻分类体系的基础上，重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。

该部分数据已经经过处理，放在了data/data59734下。如果有想了解原始数据的同学，可以去官网查询。

训练过程中所需要的预训练模型在data/data56340下。

ok，到这里我们有关BERT的课程就基本结束了，最后留给大家一个代码作业。

到这里，大家可以启动GPU环境来完成作业了。

在work/TextClassifier-main中提供了一个基于bert的baseline，大家针对下面要求完成作业就好。

**作业提交要求：**
1. 修改baseline，利用前面课程中提出的任何一种方法（用cnn等改造微调模型、调参、改变迁移策略等等），并跑至少4个epoch。同时将print的结果图片发到这里（本文最后我留一行让大家加图片）。
2. 将你设计的方法相关代码（或文字说明）复制到我预留的位置，方便老师查阅。

## 7.1 训练过程中的注意事项

1.原始数据大概要35w条，为了缩短计算时间，如下如所示，我将数据做了5w条的采样。大家如果想用全量数据试验，可以自行修改代码。

![](https://ai-studio-static-online.cdn.bcebos.com/0d2ae688ada64edea7a375ba06e34c536e8dcbff859848dbb80db26a5c546a7d)


2.训练过程中需要查看GPU使用情况，可以如下图所示打开一个新的终端，并在终端中执行下列代码。


```python
watch -n 0.1 -d nvidia-smi
```

![](https://ai-studio-static-online.cdn.bcebos.com/d3b532f50a494ef7b045a087c1844fa7fcf43bce94804bb3a18a7c3bd2cf06c1)

3.下图就是大家需要提交自己训练结果的截图实例。

<img src="https://ai-studio-static-online.cdn.bcebos.com/c2e4d51a0c4c474ba2db13d1a8d20e76541746144ba64b86966e326700ea919d" width="800" />

## 7.2 提交作业处

1.至少训练4个epoch以后的结果截图（使用数据量不限），请添加到下面一栏

你的图

2.你的代码或者做法解释，请添加在下面一栏

code或者文字


```python

```
