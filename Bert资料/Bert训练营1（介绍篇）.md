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

# 1 BERT模型的初步认识

同学们大家好，我们的课程就正式开始了啊！ 不知道BERT（Pre-training of Deep Bidirectional Transformers，原文链接：BERT）没有关系，让我们来看一些数据。

自从2018年BERT模型发布以来，BERT模型仅用 2019 年一年的时间，便以势如破竹的姿态成为了 NLP 领域首屈一指的红人，BERT 相关的论文也如涌潮般发表出来。2019 年，可谓是 NLP 发展历程中具有里程碑意义的一年，而其背后的最大功臣当属 BERT ！在NLP领域，如果把2019年称为“BERT年”也不为过。 据统计，2019以BERT为主要内容的论文发表数量近200篇，具体数据可以看看下面图片的github链接呦。

<img src="https://ai-studio-static-online.cdn.bcebos.com/a9ac464e38a44044a65b33bf4e1ae99253e712727d974c9c9e4af65032f96733" width="800" />

图片出处：[BERT_papers](http://github.com/nslatysheva/BERT_papers)，点点看，里面有彩蛋。

# 2 BERT发布之前NLP的状态

在BERT模型发布之前，NLP任务的解决方案是基于word2vec这样的词特征+RNNs等网络结构的解决方案。由于RNNs模型的特征提取能力不足，为了满足业务指标，往往需要大量的标注数据才能满足上线需求。这样，小一些的NLP公司，由于相对数据的匮乏，难以推动NLP业务，致使NLP技术的发展并没有像计算机视觉领域那么顺利。不过，自然语言处理和计算机视觉的研究本身就是互相交叉互相影响的，所以就有学者基于图像领域的思想，将NLP问题的思路转化为预训练+微调的模式，取得了优异的成绩，在BERT模型发表之前，ELMo、GPT就是这一模式的典型开创者。

ELMo提出的预训练+微调的模式，当时只是为了解决word2vec词向量不能表达”一词多义“的问题提出来的。它是一种动态词向量的思想，不过这种预训练+微调的模式借助迁移学习的思想为后来BERT的出现打下了一定的基础，本章不会具体阐述ELMo的原理，如果你还不了解ELMo，我强烈建议你阅读下面的材料来补充相关的知识。

ELMo相关阅读资料：（[ELMo原理解析及简单上手使用](https://zhuanlan.zhihu.com/p/51679783)，不管你看没看，都要问一下自己下面这道思考题啊！当然，阅读资料里也能找到。😁 ）



为了解决上述问题，2017年Google发布了特征提取能力更为强大的Transformer网络，论文链接：[Attention is all you need](http://arxiv.org/abs/1706.03762)。Transformer中的具体结构以及代码都会在后面章节结合BERT详细剖析，在此不过多介绍。

有了Transformer网络结构后，改造ELMo解决更多更难的问题，提供了一个方向。

对于近年来NLP领域模型发展的历史可以观看下图，该图出自ACL2019大会报告（The Bright Future of ACL/NLP，可以在课程中下载）。

![](https://ai-studio-static-online.cdn.bcebos.com/da206850e9694cbb9e21af24cb0d7e50617565e980e64240a2bcc3bc0c5ef851)

不过话说回来，第一个使用Transformer的预训练模型不是bert，而是GPT。想要进一步了解GPT模型的同学，可以阅读补充资料（OpenAI GPT: Generative Pre-Training for Language Understanding），如果你不了解Transformer结构，我建议你先不要阅读，等学完后面课程BERT中Transformer的细节后，再来品味一下GPT与BERT的不同。

提前透露一下GPT和BERT的最大的不同，GPT里的基本结构是由单向的Transformer-Decoder结构组成，而BERT是由双向的Transformer-Encoder结构组成。

不管是ELMo、GPT还是BERT，他们改变解决NLP的重要思路是预训练+微调的模式。如图所示，预训练+微调的模式是一种迁移学习的思想，预训练阶段可以使用大规模的数据（比如wiki），使得一个强大的模型学习到大量的知识，而且这些知识的学习方式是无监督的。通过预训练的学习之后，模型就已经具备大量的先验知识，在微调阶段继续使用预训练好的模型参数，采用业务自身标注数据在此基础上完成最后一步的监督学习。

<img src="https://ai-studio-static-online.cdn.bcebos.com/4f8d130b390b4d5a82f065ff57029c9131e7dd7beb9b445da0dfc01cdeb3d7ba" width="500" />

# 3 带你读BERT原论文

考验大家的环节到了啊，我们需要精读一下BERT的原文，我也已经将重要的信息做了标注和解释，大家打开来阅读吧（退回课程主页，打开有标注的BERT原文）。

把标注的地方全部弄清楚，就可以进行下面的学习了，如果有什么疑问可以在群里咨询啊，一个新的算法，理论啃完，在把源码吃透，才算真的掌握，希望大家多注意细节。



<img src="https://ai-studio-static-online.cdn.bcebos.com/95d0106a8b3e43bfba4ddfb8ff618366d1ff9c9324254e71af6ca7cbd7ad4496" width="600" />

# 4 BERT发布时的成绩

BERT当年发表时就在SQuAD v1.1上，获得了93.2％的 F1 分数，超过了之前最高水准的分数91.6％，同时超过了人类的分数91.2%。

<img src="https://ai-studio-static-online.cdn.bcebos.com/318a6d6718ac45c796864e08cd89151a5b92955427f04be084d22fef9583eacb" width="700" />

BERT 还在极具挑战性的 GLUE 基准测试中将准确性的标准提高了7.6％。这个基准测试包含 9 种不同的自然语言理解（NLU）任务。在这些任务中，具有人类标签的训练数据跨度从 2,500 个样本到 400,000 个样本不等。BERT 在所有任务中都大大提高了准确性。

<img src="https://ai-studio-static-online.cdn.bcebos.com/8729ae9d54b74236aa56932a2e282ad0b50d18ef8eae402c802ae2d1fb3b8e8b" width="800" />

上述的微调任务介绍如下表

- MNLI：给定一个前提 (Premise) ，根据这个前提去推断假设 (Hypothesis) 与前提的关系。该任务的关系分为三种，蕴含关系 (Entailment)、矛盾关系 (Contradiction) 以及中立关系 (Neutral)。所以这个问题本质上是一个分类问题，我们需要做的是去发掘前提和假设这两个句子对之间的交互信息。
- QQP：基于Quora，判断 Quora 上的两个问题句是否表示的是一样的意思。
- QNLI：用于判断文本是否包含问题的答案，类似于我们做阅读理解定位问题所在的段落。
- STS-B：预测两个句子的相似性，包括5个级别。
- MRPC：也是判断两个句子是否是等价的。
- RTE：类似于MNLI，但是只是对蕴含关系的二分类判断，而且数据集更小。
- SWAG：从四个句子中选择为可能为前句下文的那个。
- SST-2：电影评价的情感分析。
- CoLA：句子语义判断，是否是可接受的（Acceptable）。

# 5 BERT模型的核心架构

通过上面章节的学习，大家对BERT应该有了基本的认识。在运行最后一段代码时应该已经发现，我们采用了[Transformers预训练模型库](https://github.com/huggingface/transformers)来实现BERT的功能，所以我们这节课的代码依然以此为基础。

在work文件夹下已经为大家准备了transformers库的压缩包，可以在终端中自行解压。

从理论的角度看，想要了解BERT的模型结构，需要补充Transformer（以自注意力为主）结构的相关知识，论文链接上节已经给出。不过BERT并没有采用整个的Transformer结构，仅使用了Transformer结构里的Encoder部分。BERT将多层的Encoder搭建一起组成了它的基本网络结构，本节课我们会从BERT的源代码角度分析BERT的核心。

下面我们看看整个的BERT模型是什么样的，大体结构如下图所示。

<img src="https://ai-studio-static-online.cdn.bcebos.com/dc50f60f212f4f7ea0c2a490179e8795323399aa27414b31a1f825a51f044909" width="500" />

## 5.1 BertEncoder

红色框圈出的部分就是BERT的整个核心网络结构，我们管他叫做BERT Encoder，该部分代码在work/transformers/src/transformers/modeling_bert.py中（transformers解压后路径），相应的类是BertEncoder，代码如下。

本节课采用的源码出自Transformers 3.4.0，同学可以自己去github上pull，也可以看work目录下的代码


```python
import torch
from torch import nn
```


```python
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 由多层BertLayer（Transformer Encoder）组成，论文中给出，bert-base是12层，bert-large是24层，一层结构就如上图中蓝色框里的结构
        # config.num_hidden_layers) = 12 or 24
        # nn.ModuleList称之为容器，使用方法和python里的list类似
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
```

## 5.2 BertLayer

<img src="https://ai-studio-static-online.cdn.bcebos.com/e3f69ac3c0d24bfa930bf46006c5ffee247ef51ed48b46d5b9bc697fc409aad9" width="700" />

通过上面的代码能够看出，BertEncoder是由多层BertLayer叠加而成，我们把BertLayer这个结构单独拿出来，如上图中右半部分所示。它的代码结构如下。


```python
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 这句话是最新模型Reformer用来降低FFN内存占用空间大的问题，BERT模型不用考虑
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # Reformer才会用，此处不用
        self.seq_len_dim = 1
        # BertAttention为图中的Multi-Head Attention结构+其上面第一个Add & Norm结构
        self.attention = BertAttention(config)
        # 如果需要有用到Transformer的decoder结构，就需要跑下面11-14行代码，但纯BERT模型是不要的
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            self.crossattention = BertAttention(config)
        # 图中的Feed Forward结构
        self.intermediate = BertIntermediate(config)
        # 图中Feed Forward结构上面的Add & Norm结构
        self.output = BertOutput(config)
```

BertLayer类的初始定义上面已经给出，下面通过BertLayer类中的forward函数看下整个模型的运行流程。

<img src="https://ai-studio-static-online.cdn.bcebos.com/e30904ae33d94e7cb50a7d6975aa36f65f4729718156449cbc78193ebd848237" width="600" />


```python
def forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    output_attentions=False,
):
    # BertAttention，传参进去的mask后续会有详细解释
    self_attention_outputs = self.attention(
        hidden_states,
        attention_mask,
        head_mask,
        output_attentions=output_attentions,
    )
    # self_attention_outputs[0]是通过多头自注意力后的输出结果
    attention_output = self_attention_outputs[0]
    # self_attention_outputs[1]是子注意力计算过程中产生的attention weights
    outputs = self_attention_outputs[1:]
    
    """
    注释掉的这段代码在BERT里是不用的，在一些用于生成式任务的预训练模型会使用，其实这个地方是能体现出BERT和GPT的不同。

    if self.is_decoder and encoder_hidden_states is not None:
        assert hasattr(
            self, "crossattention"
        ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights
    """
    # 这段代码里的chunking部分不是给BERT用的，但是源码把BertIntermediate、BertOutput都封装到里面了，所以我们直接看feed_forward_chunk这个函数就可以了
    layer_output = apply_chunking_to_forward(
        self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
    )
    outputs = (layer_output,) + outputs
    return outputs

def feed_forward_chunk(self, attention_output):
    # BertIntermediate，结构见图
    intermediate_output = self.intermediate(attention_output)
    # BertOutput，结构见图
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output
```

## 5.3 BertAttention

拆到这里不能停，我们还要继续拆解，找到BERT最核心的代码。

我们继续采用倒序拆解的步骤分析模型的源码，BertAttention可以拆分BertSelfAttention和BertSelfOutput两部分，如下所示。


```python
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        # pruned_heads操作很有意思，此处可以扩展阅读，见下面扩展。如果仅了解BERT最基本的，可以不用看。
        self.pruned_heads = set()
```

扩展阅读：Transformer中multi-head的每个head到底在干嘛？如果有打酱油的head是否可以直接丢掉？具体可以延伸阅读[Are Sixteen Heads Really Better than One?](http://arxiv.org/pdf/1905.10650.pdf)。

好的，到这位置，最为重要的self attention（自注意力）结构终于出现了，大家要仔细阅读啊，我们先看看自注意力的原理图。

<img src="https://ai-studio-static-online.cdn.bcebos.com/d899c3a39d4343afb3402e9e2ea70673e3d068625bbc4450aae53f7907cb79d4" width="600" />

Self-Attention在整个Transformer结构中是最重要的基本结构单元，整个计算过程围绕着一个公式展开。
$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
K、Q、V三个向量是在训练过程中通过token的embedding与三个不同的权重矩阵分别相乘得到的，通过Self-Attention的计算过程后完成上图左半边的结构。

下图拿Thinking Machines一个句子展示了整个的自注意力计算过程。

<img src="https://ai-studio-static-online.cdn.bcebos.com/46871d1ea8844adfbc9ac21612499b4d86493bf9a3db4b1393ac19a5c6ad0dcf" width="600" />

上图出处：[http://jalammar.github.io/illustrated-transformer/](http://jalammar.github.io/illustrated-transformer/)，因为上图只是一步计算的展示，整个自注意力计算是需要对每个token都进行一遍。关于整个过程的形象展示，大家直接精读这篇博客就可以了。



```python
import math

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # hidden_size是能整除头数的，否则就会报错，BERT-base里hidden_size为768
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        # 多头的头数
        self.num_attention_heads = config.num_attention_heads
        # 每个头的维度
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            # 同学们想一想这一步加的mask的作用？
            attention_scores = attention_scores + attention_mask 

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs
```


```python
class Config:
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.attention_probs_dropout_prob = 0.1

config = Config()
bertSelfAttention = BertSelfAttention(config)
input_tensor = torch.ones([2, 5, 768])
output_tensor = bertSelfAttention(input_tensor)
print(output_tensor)
```

![](https://ai-studio-static-online.cdn.bcebos.com/7fcd9d9ff30344f6ba42027540b10239e49dae940a5e4a8987876845c5d13c5a)

BERT中最重要的模型细节，我们就拆解完了，其他的代码就比较好理解了，大家稍微回顾下上图的过程，就可以继续了。

看完BertSelfAttention后，我们看下BertSelfOutput到底是什么呢，其实它就是BertSelfAttention上面的Add & Norm层，它是采用了残差结构并进行层级归一化的操作。

关于残差结构，它主要是解决神经网络梯度消失的现象，可以参考cv领域中的一些解释，如[为什么ResNet和DenseNet可以这么深？一文详解残差块为何有助于解决梯度弥散问题](https://zhuanlan.zhihu.com/p/28124810)。具体的操作如下面代码。
```
hidden_states = self.LayerNorm(hidden_states + input_tensor)
# hidden_states + input_tensor实现了一个残差的操作
```
除了残差结构，同时增加了一个LayerNorm实现层级归一化操作，关于LayerNorm，可以研究下[详解深度学习中的Normalization，BN/LN/WN](https://zhuanlan.zhihu.com/p/33173246)。



```python
# BertSelfOutput代码如下，可以看下具体是怎么做的Add & Norm
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```

## 5.4 Feed-Forward Network

Encoder中存在的另一个结构是前馈神经网络，也就是Feed-Forward Network，它的作用是加深我们的网络结构。FFN层包括两个线性操作，中间有一个 ReLU 激活函数，对应到公式的形式为：

$$FFN(x) = \max(0, xW_1+b_1)W_2 + b2$$

其实，FFN的加入引入了非线性(ReLu等激活函数)，变换了attention output的空间, 从而增加了模型的表现能力。把FFN去掉模型也是可以用的，但是效果差了很多。

代码见BertIntermediate类。


```python
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            # ACT2FN是一个可配置的激活函数，有RELU等
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        # 第一层全连接
        hidden_states = self.dense(hidden_states)
        # 第一层全连接的非线性激活函数，第二层全连接做到了BertOutput里
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
```

BertIntermediate层上面是最后一层BertOutput，主要是为了完成FFN的第二层全连接操作和最后的Add & Norm。


```python
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # FFN的第二层线性全连接
        hidden_states = self.dense(hidden_states)
        # 实际代码中会添加dropout防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 最后的Add & Norm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```

## 5.5 Mask

除了Encoder的整体代码之外，贯穿始终的还有一个细节，就是代码中有很多的Mask操作，那他们都是做什么的呢？

在work/transformers/src/transformers/modeling_bert.py中可以查到class BertModel，其中

```
self.encoder = BertEncoder(config)
```
我们可以看到存在很多的mask。


```python
encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
```

Mask表示掩码，它是对某些值进行掩盖，让其不参加计算。Transformer模型中涉及到两种mask，一种是padding mask，一种是sequence mask只存在decoder中。。padding mask是在encoder和decoder中都存在的，而sequence mask只存在decoder中。

对于这两种mask是怎么计算的，可以参照work/transformers/src/transformers/modeling_utils.py中的get_extended_attention_mask函数。

部分代码见下面。


```python
def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder: # bert不涉及decoder，如果需要decoder时是要考虑这里的mask
            batch_size, seq_length = input_shape
            seq_ids = torch.arange(seq_length, device=device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            # in case past_key_values are used we need to add a prefix ones mask to the causal mask
            # causal and attention masks must have same type with pytorch version < 1.3
            causal_mask = causal_mask.to(attention_mask.dtype)

            if causal_mask.shape[1] < attention_mask.shape[1]:
                prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                causal_mask = torch.cat(
                    [
                        torch.ones(
                            (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                        ),
                        causal_mask,
                    ],
                    axis=-1,
                )

            extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                input_shape, attention_mask.shape
            )
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask
```

代码比较复杂，不过原理很简单。

padding mask，在nlp任务中由于一个batch数据中句子长短不一，所以需要对一些句子进行padding，而这些padding的数据在后面自注意力计算上是没有意义的，所以在计算中要忽略padding的影响。

具体实现是见源码中这段
```
extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
```
把padding位置上的值加上一个非常大的负数，这样经过softmax计算后这个位置的概论无限接近零。

sequence mask，在做生成问题时，为了不让模型看到后面的标准答案，所以在预测t位置词的时候，把t后面所有的词都mask掉，具体实现是通过一个对角矩阵实现的，此处就不细讲了。注意由于decoder过程中padding mask和sequence mask都存在，所以上面源码有两个mask想加的过程。

好的，整个BERT核心源码已经讲完了，如果同学们在理论的细节上还是有一些模糊，可以补充阅读[Transformer模型深度解读](https://zhuanlan.zhihu.com/p/104393915)，该博客从Transformer的角度详细做了解释，看完后，回头在看看BERT的源码，我相信理解会进一步加深。

# 6 模型输入之Embedding层

BERT模型的输入不管是单个句子还是多个句子，都会将句子中的Token转化成Embedding之后传到模型里，那BERT的Embedding是怎么做的呢？它是由Token Embeddings、Segment Embeddings和Position Embeddings想加组成，如下图。

其中：

1）Token Embeddings是词向量，是将Token嵌入到一个维度的空间，BERT随着结构层数的变化，分别选取了768维和1024维。在Token的输入上，BERT也做了特殊的处理，第一个Token是一个CLS的特殊字符，可以用于下游的任务。句子和句子的中间，以及句子的末尾会有一个特殊的符号SEP；

2）Segment Embeddings用来区别两种句子，因为预训练不光做LM还要做以两个句子为输入的分类任务；

3）Position Embeddings和上一章的Transformer不一样，不是三角函数而是一个跟着训练学出来的向量。

<img src="https://ai-studio-static-online.cdn.bcebos.com/8c8dc740398c4bb597fdf40b37f3443e8344dc0f65b4493984adde5fac6296ed" width="700" />


```python
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
```

以上代码就是BERT Embedding层的一些定义，最终会将三种Embedding想加。

```
embeddings = inputs_embeds + position_embeddings + token_type_embeddings
```
同学们是不是还会有一些疑惑，为什么BERT的位置编码这么简单粗暴，关于一些相关探讨，可以阅读[如何优雅地编码文本中的位置信息？三种positional encoding方法简述](https://zhuanlan.zhihu.com/p/121126531)。

# 7 模型输出层与CLS

关于模型的输出层，大家可以参照一下源码。具体的使用，以及其中的一些特殊符号都会在后续课程中详解。


```python
class BertPooler(nn.Module):
    # Pooler拿的是第一个token的最终输出，也就是CLS特殊符号的输出
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

至此有关BERT的核心代码内容介绍就结束了，大家结合课上的内容，多多研究源码，对BERT掌握会更上一层楼。

# 8 作业

## 8.1作业要求

> 作业一

ELMo的缺点是什么呢？

> 作业二

BERT采用了迁移学习的思想，如果在相同的NLP任务上达到传统模型（如RNN等）一样的性能指标，比如准确度都是90%，在准备数据成本上有优势么，什么样的优势？

> 作业三

问个问题，多头注意力是如何实现的，具体是在哪个tensor切分的头，为什么要多头？

> 作业四

为什么BERT里面用的是LN，而不是BN？

## 8.2作业提交处

作业一

作业二

作业三

作业四


```python

```
