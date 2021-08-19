import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import BertConfig, BertForMaskedLM
from transformers import DistilBertConfig, DistilBertForMaskedLM

class PORT(nn.Module):
    def __init__(self, args, output_hidden_states=True, output_attentions=True):
        super(PORT, self).__init__()
        config = BertConfig()
        config.hidden_size = args.embd_dim
        config.num_attention_heads = args.n_heads
        config.num_hidden_layers = args.n_layers
        config.intermediate_size = args.hidden_dim
        config.dropout_prob = args.dropout_port
        config.return_dict = True
        config.output_attentions = output_attentions
        config.output_hidden_states = output_hidden_states
        config.max_position_embeddings = 16
        config.position_embedding_type = 'absolute'
        config.vocab_size = 3

        self.pose_projection = nn.Linear(3, config.hidden_size)
        self.bert = BertForMaskedLM(config)

    def forward(self, x):
        embeds = self.pose_projection(x)
        return self.bert(inputs_embeds=embeds)

