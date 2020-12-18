import torch.nn as nn
from transformers import BertModel, BertConfig
import torch.nn.functional as F
import torch

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

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.num_labels = config.num_classes
        model_config = BertConfig.from_pretrained(
            config.bert_path, num_labels=config.num_classes)
        self.bert = BertModel.from_pretrained(config.bert_path,
                                              config=model_config)
        self.dropout = nn.Dropout(config.dropout)
        self.convs = Conv1d(config.hidden_size, config.num_filters, config.filter_sizes)
        self.classifier = nn.Linear(len(config.filter_sizes) * config.num_filters,self.num_labels)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        token_type_ids = x[2]
        
        encoded_layers, _ = self.bert(context,
                              attention_mask=mask,
                              token_type_ids=token_type_ids)
        # encoded_layers: [batch_size, seq_len, bert_dim=768]
        encoded_layers = self.dropout(encoded_layers)

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
        return logits

'''
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        model_config = BertConfig.from_pretrained(
            config.bert_path, num_labels=config.num_classes)
        self.bert = BertModel.from_pretrained(config.bert_path,
                                              config=model_config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[1]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        token_type_ids = x[2]
        _, pooled = self.bert(context,
                              attention_mask=mask,
                              token_type_ids=token_type_ids)
        out = self.fc(pooled)
        return out

'''