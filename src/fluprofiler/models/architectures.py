"""
Model architectures for fluProfiler.

Contains the main model classes including various fluProfiler variants.
"""

import torch
import copy
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.configuration_utils import PretrainedConfig

from .config import fluProfiler_Config
from .losses import create_loss_function
from .pooling import value_pooling, attention_mask, attention_pooling, value_weighting




class fluProfiler_v0_1(PreTrainedModel):
    """ 
    Original fluProfiler model v0.1
    """
    config_class = fluProfiler_Config

    def __init__(self, config, args):
        super(fluProfiler_v0_1, self).__init__(config)
        if config.seq_max_length is None and config.seq_max_length_a == config.seq_max_length_b == config.seq_max_length_c == config.seq_max_length_d:
            config.seq_max_length = config.seq_max_length_a
        if config.matrix_max_length is None and config.matrix_max_length_a == config.matrix_max_length_b == config.matrix_max_length_c == config.matrix_max_length_d:
            config.matrix_max_length = config.matrix_max_length_a
        if config.embedding_input_size is None and config.embedding_input_size_a == config.embedding_input_size_b == config.embedding_input_size_c == config.embedding_input_size_d:
            config.embedding_input_size = config.embedding_input_size_a

        self.num_labels = config.num_labels
        self.fusion_type = args.fusion_type if hasattr(args, "fusion_type") and args.fusion_type else "concat"
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type
        self.prepend_bos = args.prepend_bos
        self.append_eos = args.append_eos

        self.seq_encoder, self.seq_pooler, \
        self.matrix_encoder, self.matrix_pooler = None, None, None, None
        self.encoder_type_list = [False, False, False]
        self.input_size_list = [0, 0, 0]
        self.linear_idx = [-1, -1, -1]

        self.matrix_dropout = nn.Dropout(p=0.1)

        self.input_size_list[1] = config.hidden_size

        new_config = copy.deepcopy(config)
        new_config.embedding_input_size = config.hidden_size
        self.matrix_pooler = value_pooling(embed_size=new_config.embedding_input_size)


        self.encoder_type_list[1] = True
        self.linear_idx[1] = 0


        fc_size_list = [config.seq_fc_size, config.matrix_fc_size, config.vector_fc_size]
        all_linear_list = [None, None, None]
        self.output_size = [0, 0, 0]
        print("self.encoder_type_list:", self.encoder_type_list)
        for encoder_idx, encoder_flag in enumerate(self.encoder_type_list):
            if not encoder_flag:
                continue
            fc_size = fc_size_list[encoder_idx]
            input_size = self.input_size_list[encoder_idx]
            print("encoder_idx", encoder_idx, "input_size:", input_size)
            if fc_size is not None and len(fc_size) > 0:
                if isinstance(fc_size, list):
                    fc_size = [int(v) for v in fc_size]
                else:
                    fc_size = [int(fc_size)]
                linear_list = []
                for idx in range(len(fc_size)):
                    linear = nn.Linear(input_size, fc_size[idx])
                    linear_list.append(linear)
                    linear_list.append(nn.Tanh())
                    input_size = fc_size[idx]
                all_linear_list[encoder_idx] = nn.ModuleList(linear_list)
                self.output_size[encoder_idx] = fc_size[-1]
            else:
                # 没有全连接层
                self.linear_idx[encoder_idx] = -1
                self.output_size[encoder_idx] = input_size
        all_linear_list = [linear for linear in all_linear_list if linear is not None]
        if all_linear_list is not None and len(all_linear_list) > 0:
            self.linear = nn.ModuleList(all_linear_list)

        last_hidden_size = sum(self.output_size)

        last_hidden_size = last_hidden_size
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(
                config,
                args,
                hidden_size=last_hidden_size * 6 + 256,
                classifier_size=args.classifier_size,
                sigmoid=args.sigmoid,
                output_mode=args.output_mode,
                num_labels=self.num_labels,
                loss_type=args.loss_type,
                ignore_index=args.ignore_index,
                return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
            )

        self.Passage_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=6, embedding_dim=256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def __forward__(
            self,
            matrices,
            matrix_attention_masks,
            save_attention_path
    ):
        if matrices is not None:
            matrices = self.matrix_dropout(matrices)
            matrix_vector = self.matrix_pooler(matrices, mask=matrix_attention_masks, save_attention_path=save_attention_path)

            matrix_linear_idx = self.linear_idx[1]
            if matrix_linear_idx != -1:
                for i, layer_module in enumerate(self.linear[matrix_linear_idx]):
                    matrix_vector = layer_module(matrix_vector)

        concat_vector = matrix_vector

        return concat_vector

    def forward(
            self,
            input_ids_a=None, input_ids_b=None, input_ids_c=None, input_ids_d=None,
            position_ids_a=None, position_ids_b=None, position_ids_c=None, position_ids_d=None,
            token_type_ids_a=None, token_type_ids_b=None, token_type_ids_c=None, token_type_ids_d=None,
            seq_attention_masks_a=None, seq_attention_masks_b=None, seq_attention_masks_c=None, seq_attention_masks_d=None,
            vectors_a=None, vectors_b=None, vectors_c=None, vectors_d=None,
            matrices_a=None, matrices_b=None, matrices_c=None, matrices_d=None,
            matrix_attention_masks_a=None, matrix_attention_masks_b=None, matrix_attention_masks_c=None, matrix_attention_masks_d=None,
            strainPassCats=None, labels=None,
            save_concat_vector=None,
            save_attention_path=None,
            **kwargs
    ):
        representation_vector_a = self.__forward__(
            matrices_a,
            matrix_attention_masks_a,
            save_attention_path
        )

        representation_vector_b = self.__forward__(
            matrices_b,
            matrix_attention_masks_b,
            save_attention_path
        )

        representation_vector_c = self.__forward__(
            matrices_c,
            matrix_attention_masks_c,
            save_attention_path
        )

        representation_vector_d = self.__forward__(
            matrices_d,
            matrix_attention_masks_d,
            save_attention_path
        )

        strainPassCats_vector = torch.mean(self.Passage_encoder(strainPassCats), dim=1)

        diff_ac = representation_vector_a - representation_vector_c   # 你想要的 a-c
        prod_ac = representation_vector_a * representation_vector_c   # 逐元素乘

        concat_vector = torch.concat([representation_vector_a, representation_vector_b, representation_vector_c, representation_vector_d, diff_ac, prod_ac], dim=1)

        if save_concat_vector is not None:
            filenames = os.listdir(save_concat_vector)
            if len(filenames):
                max_num = max([int(filename.split('.pth')[0]) for filename in filenames])
                torch.save(concat_vector, save_concat_vector + "/{}.pth".format(max_num+1))
            else:
                torch.save(concat_vector, save_concat_vector + "/1.pth")

        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)

        concat_vector = torch.concat([concat_vector, strainPassCats_vector], dim=1)

        concat_vector = self.hidden_layer(concat_vector)
        concat_vector = self.hidden_act(concat_vector)

        logits = self.classifier(concat_vector)
        output = logits

        outputs = [logits, output]
        if labels is not None:
            if self.output_mode in ["regression"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N, seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels)
                else:
                    # logits: N * seq_len
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N ,seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # logits: N * seq_len * 1
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1).float())

            outputs = [loss, *outputs]
        return outputs

class fluProfiler_v3_0(PreTrainedModel):
    """ 
    这个版本比v0_1做了以下改进：
    1. 仅对HA序列做reweight，对NA直接进行max pooling
    2. reweight的时候加入残差链接，避免对原始基模的特征破坏(暂时没放)
    """
    config_class = fluProfiler_Config

    def __init__(self, config, args):
        super(fluProfiler_v3_0, self).__init__(config)
        if config.seq_max_length is None and config.seq_max_length_a == config.seq_max_length_b == config.seq_max_length_c == config.seq_max_length_d:
            config.seq_max_length = config.seq_max_length_a
        if config.matrix_max_length is None and config.matrix_max_length_a == config.matrix_max_length_b == config.matrix_max_length_c == config.matrix_max_length_d:
            config.matrix_max_length = config.matrix_max_length_a
        if config.embedding_input_size is None and config.embedding_input_size_a == config.embedding_input_size_b == config.embedding_input_size_c == config.embedding_input_size_d:
            config.embedding_input_size = config.embedding_input_size_a

        self.num_labels = config.num_labels
        self.fusion_type = args.fusion_type if hasattr(args, "fusion_type") and args.fusion_type else "concat"
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type
        self.prepend_bos = args.prepend_bos
        self.append_eos = args.append_eos

        self.seq_encoder, self.seq_pooler, \
        self.matrix_encoder, self.matrix_pooler = None, None, None, None
        self.encoder_type_list = [False, False, False]
        self.input_size_list = [0, 0, 0]
        self.linear_idx = [-1, -1, -1]

        self.matrix_dropout = nn.Dropout(p=0.1)

        self.input_size_list[1] = config.hidden_size

        new_config = copy.deepcopy(config)
        new_config.embedding_input_size = config.hidden_size
        self.matrix_pooler = value_pooling(embed_size=new_config.embedding_input_size)


        self.encoder_type_list[1] = True
        self.linear_idx[1] = 0


        fc_size_list = [config.seq_fc_size, config.matrix_fc_size, config.vector_fc_size]
        all_linear_list = [None, None, None]
        self.output_size = [0, 0, 0]
        print("self.encoder_type_list:", self.encoder_type_list)
        for encoder_idx, encoder_flag in enumerate(self.encoder_type_list):
            if not encoder_flag:
                continue
            fc_size = fc_size_list[encoder_idx]
            input_size = self.input_size_list[encoder_idx]
            print("encoder_idx", encoder_idx, "input_size:", input_size)
            if fc_size is not None and len(fc_size) > 0:
                if isinstance(fc_size, list):
                    fc_size = [int(v) for v in fc_size]
                else:
                    fc_size = [int(fc_size)]
                linear_list = []
                for idx in range(len(fc_size)):
                    linear = nn.Linear(input_size, fc_size[idx])
                    linear_list.append(linear)
                    linear_list.append(nn.Tanh())
                    input_size = fc_size[idx]
                all_linear_list[encoder_idx] = nn.ModuleList(linear_list)
                self.output_size[encoder_idx] = fc_size[-1]
            else:
                # 没有全连接层
                self.linear_idx[encoder_idx] = -1
                self.output_size[encoder_idx] = input_size
        all_linear_list = [linear for linear in all_linear_list if linear is not None]
        if all_linear_list is not None and len(all_linear_list) > 0:
            self.linear = nn.ModuleList(all_linear_list)

        last_hidden_size = sum(self.output_size)

        # v3_0: a,c 走 __forward__ 输出 last_hidden_size(256)，b,d 直接 max pooling 输出 config.hidden_size(2560)
        # concat = a+b+c+d+diff_ac+prod_ac+strainPass = 256+2560+256+2560+256+256+256 = 6400
        v3_0_hidden_size = 4 * last_hidden_size + 2 * config.hidden_size + 256
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(
                config,
                args,
                hidden_size=v3_0_hidden_size,
                classifier_size=args.classifier_size,
                sigmoid=args.sigmoid,
                output_mode=args.output_mode,
                num_labels=self.num_labels,
                loss_type=args.loss_type,
                ignore_index=args.ignore_index,
                return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
            )

        self.Passage_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=6, embedding_dim=256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def __forward__(
            self,
            matrices,
            matrix_attention_masks,
            save_attention_path
    ):
        if matrices is not None:
            matrices = self.matrix_dropout(matrices)
            matrix_vector = self.matrix_pooler(matrices, mask=matrix_attention_masks, save_attention_path=save_attention_path)

            matrix_linear_idx = self.linear_idx[1]
            if matrix_linear_idx != -1:
                for i, layer_module in enumerate(self.linear[matrix_linear_idx]):
                    matrix_vector = layer_module(matrix_vector)

        concat_vector = matrix_vector

        return concat_vector

    def forward(
            self,
            input_ids_a=None, input_ids_b=None, input_ids_c=None, input_ids_d=None,
            position_ids_a=None, position_ids_b=None, position_ids_c=None, position_ids_d=None,
            token_type_ids_a=None, token_type_ids_b=None, token_type_ids_c=None, token_type_ids_d=None,
            seq_attention_masks_a=None, seq_attention_masks_b=None, seq_attention_masks_c=None, seq_attention_masks_d=None,
            vectors_a=None, vectors_b=None, vectors_c=None, vectors_d=None,
            matrices_a=None, matrices_b=None, matrices_c=None, matrices_d=None,
            matrix_attention_masks_a=None, matrix_attention_masks_b=None, matrix_attention_masks_c=None, matrix_attention_masks_d=None,
            strainPassCats=None, labels=None,
            save_concat_vector=None,
            save_attention_path=None,
            **kwargs
    ):
        representation_vector_a = self.__forward__(
            matrices_a,
            matrix_attention_masks_a,
            save_attention_path
        )

        # NA 序列（b）：不再送入 __forward__，而是直接做 max pooling
        mask = matrix_attention_masks_b.unsqueeze(-1).to(dtype=matrices_b.dtype)  # (B, L, 1)
        masked = matrices_b.masked_fill(mask == 0, torch.finfo(matrices_b.dtype).min)
        representation_vector_b = masked.max(dim=1).values  # (B, E)

        representation_vector_c = self.__forward__(
            matrices_c,
            matrix_attention_masks_c,
            save_attention_path
        )

        mask = matrix_attention_masks_d.unsqueeze(-1).to(dtype=matrices_d.dtype)  # (B, L, 1)
        masked = matrices_d.masked_fill(mask == 0, torch.finfo(matrices_d.dtype).min)
        representation_vector_d = masked.max(dim=1).values  # (B, E)

        strainPassCats_vector = torch.max(self.Passage_encoder(strainPassCats), dim=1).values

        diff_ac = representation_vector_a - representation_vector_c   # 你想要的 a-c
        prod_ac = representation_vector_a * representation_vector_c   # 逐元素乘

        concat_vector = torch.concat([representation_vector_a, representation_vector_b, representation_vector_c, representation_vector_d, diff_ac, prod_ac], dim=1)

        if save_concat_vector is not None:
            filenames = os.listdir(save_concat_vector)
            if len(filenames):
                max_num = max([int(filename.split('.pth')[0]) for filename in filenames])
                torch.save(concat_vector, save_concat_vector + "/{}.pth".format(max_num+1))
            else:
                torch.save(concat_vector, save_concat_vector + "/1.pth")

        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)

        concat_vector = torch.concat([concat_vector, strainPassCats_vector], dim=1)

        concat_vector = self.hidden_layer(concat_vector)
        concat_vector = self.hidden_act(concat_vector)

        logits = self.classifier(concat_vector)
        output = logits

        outputs = [logits, output]
        if labels is not None:
            if self.output_mode in ["regression"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N, seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels)
                else:
                    # logits: N * seq_len
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N ,seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # logits: N * seq_len * 1
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1).float())

            outputs = [loss, *outputs]
        return outputs

class fluProfiler_v0_1_1(PreTrainedModel):
    """ 
    Original fluProfiler model v0.1.1
    """
    config_class = fluProfiler_Config

    def __init__(self, config, args):
        super(fluProfiler_v0_1_1, self).__init__(config)
        if config.seq_max_length is None and config.seq_max_length_a == config.seq_max_length_b == config.seq_max_length_c == config.seq_max_length_d:
            config.seq_max_length = config.seq_max_length_a
        if config.matrix_max_length is None and config.matrix_max_length_a == config.matrix_max_length_b == config.matrix_max_length_c == config.matrix_max_length_d:
            config.matrix_max_length = config.matrix_max_length_a
        if config.embedding_input_size is None and config.embedding_input_size_a == config.embedding_input_size_b == config.embedding_input_size_c == config.embedding_input_size_d:
            config.embedding_input_size = config.embedding_input_size_a

        self.num_labels = config.num_labels
        self.fusion_type = args.fusion_type if hasattr(args, "fusion_type") and args.fusion_type else "concat"
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type
        self.prepend_bos = args.prepend_bos
        self.append_eos = args.append_eos

        self.seq_encoder, self.seq_pooler, \
        self.matrix_encoder, self.matrix_pooler = None, None, None, None
        self.encoder_type_list = [False, False, False]
        self.input_size_list = [0, 0, 0]
        self.linear_idx = [-1, -1, -1]

        self.matrix_dropout = nn.Dropout(p=0.1)

        self.input_size_list[1] = config.hidden_size

        new_config = copy.deepcopy(config)
        new_config.embedding_input_size = config.hidden_size
        self.matrix_pooler = value_pooling(embed_size=new_config.embedding_input_size)


        self.encoder_type_list[1] = True
        self.linear_idx[1] = 0


        fc_size_list = [config.seq_fc_size, config.matrix_fc_size, config.vector_fc_size]
        all_linear_list = [None, None, None]
        self.output_size = [0, 0, 0]
        print("self.encoder_type_list:", self.encoder_type_list)
        for encoder_idx, encoder_flag in enumerate(self.encoder_type_list):
            if not encoder_flag:
                continue
            fc_size = fc_size_list[encoder_idx]
            input_size = self.input_size_list[encoder_idx]
            print("encoder_idx", encoder_idx, "input_size:", input_size)
            if fc_size is not None and len(fc_size) > 0:
                if isinstance(fc_size, list):
                    fc_size = [int(v) for v in fc_size]
                else:
                    fc_size = [int(fc_size)]
                linear_list = []
                for idx in range(len(fc_size)):
                    linear = nn.Linear(input_size, fc_size[idx])
                    linear_list.append(linear)
                    linear_list.append(nn.Tanh())
                    input_size = fc_size[idx]
                all_linear_list[encoder_idx] = nn.ModuleList(linear_list)
                self.output_size[encoder_idx] = fc_size[-1]
            else:
                # 没有全连接层
                self.linear_idx[encoder_idx] = -1
                self.output_size[encoder_idx] = input_size
        all_linear_list = [linear for linear in all_linear_list if linear is not None]
        if all_linear_list is not None and len(all_linear_list) > 0:
            self.linear = nn.ModuleList(all_linear_list)

        last_hidden_size = sum(self.output_size)

        last_hidden_size = last_hidden_size
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(
                config,
                args,
                hidden_size=last_hidden_size * 6 + 16,
                classifier_size=args.classifier_size,
                sigmoid=args.sigmoid,
                output_mode=args.output_mode,
                num_labels=self.num_labels,
                loss_type=args.loss_type,
                ignore_index=args.ignore_index,
                return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
            )

        self.Passage_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=6, embedding_dim=16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )

    def __forward__(
            self,
            matrices,
            matrix_attention_masks,
            save_attention_path
    ):
        if matrices is not None:
            matrices = self.matrix_dropout(matrices)
            matrix_vector = self.matrix_pooler(matrices, mask=matrix_attention_masks, save_attention_path=save_attention_path)

            matrix_linear_idx = self.linear_idx[1]
            if matrix_linear_idx != -1:
                for i, layer_module in enumerate(self.linear[matrix_linear_idx]):
                    matrix_vector = layer_module(matrix_vector)

        concat_vector = matrix_vector

        return concat_vector

    def forward(
            self,
            input_ids_a=None, input_ids_b=None, input_ids_c=None, input_ids_d=None,
            position_ids_a=None, position_ids_b=None, position_ids_c=None, position_ids_d=None,
            token_type_ids_a=None, token_type_ids_b=None, token_type_ids_c=None, token_type_ids_d=None,
            seq_attention_masks_a=None, seq_attention_masks_b=None, seq_attention_masks_c=None, seq_attention_masks_d=None,
            vectors_a=None, vectors_b=None, vectors_c=None, vectors_d=None,
            matrices_a=None, matrices_b=None, matrices_c=None, matrices_d=None,
            matrix_attention_masks_a=None, matrix_attention_masks_b=None, matrix_attention_masks_c=None, matrix_attention_masks_d=None,
            strainPassCats=None, labels=None,
            save_concat_vector=None,
            save_attention_path=None,
            **kwargs
    ):
        representation_vector_a = self.__forward__(
            matrices_a,
            matrix_attention_masks_a,
            save_attention_path
        )

        representation_vector_b = self.__forward__(
            matrices_b,
            matrix_attention_masks_b,
            save_attention_path
        )

        representation_vector_c = self.__forward__(
            matrices_c,
            matrix_attention_masks_c,
            save_attention_path
        )

        representation_vector_d = self.__forward__(
            matrices_d,
            matrix_attention_masks_d,
            save_attention_path
        )

        strainPassCats_vector = torch.mean(self.Passage_encoder(strainPassCats), dim=1)

        diff_ac = representation_vector_a - representation_vector_c   # 你想要的 a-c
        prod_ac = representation_vector_a * representation_vector_c   # 逐元素乘

        concat_vector = torch.concat([representation_vector_a, representation_vector_b, representation_vector_c, representation_vector_d, diff_ac, prod_ac], dim=1)

        if save_concat_vector is not None:
            filenames = os.listdir(save_concat_vector)
            if len(filenames):
                max_num = max([int(filename.split('.pth')[0]) for filename in filenames])
                torch.save(concat_vector, save_concat_vector + "/{}.pth".format(max_num+1))
            else:
                torch.save(concat_vector, save_concat_vector + "/1.pth")

        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)

        concat_vector = torch.concat([concat_vector, strainPassCats_vector], dim=1)

        concat_vector = self.hidden_layer(concat_vector)
        concat_vector = self.hidden_act(concat_vector)

        logits = self.classifier(concat_vector)
        output = logits

        outputs = [logits, output]
        if labels is not None:
            if self.output_mode in ["regression"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N, seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels)
                else:
                    # logits: N * seq_len
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N ,seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # logits: N * seq_len * 1
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1).float())

            outputs = [loss, *outputs]
        return outputs


class fluProfiler_v1_2(PreTrainedModel):
    """fluProfiler model v1.2

    Design goal:
      - Down-project token embeddings (2560 -> 1280)
      - Use existing value-attention mechanism to compute token weights (no pooling)
      - Map to a new interaction space and perform cross-attention (HA: a<->c, NA: b<->d)
      - Pool after cross-attention and regress
      - Keep Passage_encoder unchanged
    """

    config_class = fluProfiler_Config

    def __init__(self, config, args):
        super().__init__(config)

        if config.seq_max_length is None and config.seq_max_length_a == config.seq_max_length_b == config.seq_max_length_c == config.seq_max_length_d:
            config.seq_max_length = config.seq_max_length_a
        if config.matrix_max_length is None and config.matrix_max_length_a == config.matrix_max_length_b == config.matrix_max_length_c == config.matrix_max_length_d:
            config.matrix_max_length = config.matrix_max_length_a
        if config.embedding_input_size is None and config.embedding_input_size_a == config.embedding_input_size_b == config.embedding_input_size_c == config.embedding_input_size_d:
            config.embedding_input_size = config.embedding_input_size_a

        self.num_labels = config.num_labels
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type

        # ---- hyperparams (with safe defaults) ----
        self.in_dim = int(getattr(config, "hidden_size", 2560))
        self.down_dim = int(getattr(config, "down_dim", 1280))
        self.cross_dim = int(getattr(config, "cross_dim", 256))
        self.cross_heads = int(getattr(config, "cross_heads", 4))
        self.dropout_prob = float(getattr(config, "dropout_prob", 0.1))

        if self.cross_dim % self.cross_heads != 0:
            raise ValueError(f"cross_dim ({self.cross_dim}) must be divisible by cross_heads ({self.cross_heads}).")

        # ---- stage 1: down-projection ----
        self.matrix_dropout = nn.Dropout(p=self.dropout_prob)
        self.down_proj = nn.Sequential(
            nn.Linear(self.in_dim, self.down_dim, bias=False),
            nn.LayerNorm(self.down_dim),
        )

        # ---- stage 2: token weighting (value-attention, no pooling) ----
        self.token_weighting = value_weighting(embed_size=self.down_dim)

        # ---- stage 3: map to cross-attention space ----
        self.to_cross = nn.Sequential(
            nn.Linear(self.down_dim, self.cross_dim, bias=False),
            nn.LayerNorm(self.cross_dim),
        )

        # ---- stage 4: cross-attention (single-layer, shared weights) ----
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.cross_dim,
            num_heads=self.cross_heads,
            dropout=self.dropout_prob,
            batch_first=True,
        )
        self.cross_ln = nn.LayerNorm(self.cross_dim)
        self.cross_drop = nn.Dropout(self.dropout_prob)

        # ---- stage 5: pooling after cross-attention ----
        self.post_pool = attention_pooling(embed_size=self.cross_dim)

        # ---- passage encoder (unchanged) ----
        self.Passage_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=6, embedding_dim=256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        # ---- head ----
        # We keep the same final feature construction as v0_1:
        # [a, b, c, d, (a-c), (a*c)] then concat passage(256)
        last_hidden_size = self.cross_dim
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = create_loss_function(
            config,
            args,
            hidden_size=last_hidden_size * 6 + 256,
            classifier_size=args.classifier_size,
            sigmoid=args.sigmoid,
            output_mode=args.output_mode,
            num_labels=self.num_labels,
            loss_type=args.loss_type,
            ignore_index=args.ignore_index,
            return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"],
        )

    @staticmethod
    def _to_key_padding_mask(mask):
        """Convert (B,L) float/int mask with 1=valid,0=pad to key_padding_mask bool with True=pad."""
        if mask is None:
            return None
        return mask == 0

    def _encode_side(self, matrices, matrix_attention_masks):
        """Down-project and compute token-wise weighting."""
        x = self.matrix_dropout(matrices)
        x = self.down_proj(x)  # (B,L,down_dim)
        x = self.token_weighting(x, mask=matrix_attention_masks)  # (B,L,down_dim)
        x = self.to_cross(x)  # (B,L,cross_dim)
        return x

    def _cross(self, q, kv, kv_mask):
        """Cross-attention with residual + layer norm."""
        key_padding_mask = self._to_key_padding_mask(kv_mask)
        out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        out = self.cross_ln(q + self.cross_drop(out))
        return out

    def _pool(self, x, mask):
        return self.post_pool(x, mask=mask)

    def forward(
        self,
        input_ids_a=None, input_ids_b=None, input_ids_c=None, input_ids_d=None,
        position_ids_a=None, position_ids_b=None, position_ids_c=None, position_ids_d=None,
        token_type_ids_a=None, token_type_ids_b=None, token_type_ids_c=None, token_type_ids_d=None,
        seq_attention_masks_a=None, seq_attention_masks_b=None, seq_attention_masks_c=None, seq_attention_masks_d=None,
        vectors_a=None, vectors_b=None, vectors_c=None, vectors_d=None,
        matrices_a=None, matrices_b=None, matrices_c=None, matrices_d=None,
        matrix_attention_masks_a=None, matrix_attention_masks_b=None, matrix_attention_masks_c=None, matrix_attention_masks_d=None,
        strainPassCats=None, labels=None,
        save_concat_vector=None,
        save_attention_path=None,
        **kwargs,
    ):
        Za = self._encode_side(matrices_a, matrix_attention_masks_a)
        Zb = self._encode_side(matrices_b, matrix_attention_masks_b)
        Zc = self._encode_side(matrices_c, matrix_attention_masks_c)
        Zd = self._encode_side(matrices_d, matrix_attention_masks_d)

        # Cross-attention in interaction space
        Zac = self._cross(Za, Zc, matrix_attention_masks_c)  # a attends c
        Zca = self._cross(Zc, Za, matrix_attention_masks_a)  # c attends a
        Zbd = self._cross(Zb, Zd, matrix_attention_masks_d)  # b attends d
        Zdb = self._cross(Zd, Zb, matrix_attention_masks_b)  # d attends b

        # Pool after cross-attention
        va = self._pool(Zac, matrix_attention_masks_a)  # (B,cross_dim)
        vb = self._pool(Zbd, matrix_attention_masks_b)
        vc = self._pool(Zca, matrix_attention_masks_c)
        vd = self._pool(Zdb, matrix_attention_masks_d)

        strainPassCats_vector = torch.mean(self.Passage_encoder(strainPassCats), dim=1)

        diff_ac = va - vc
        prod_ac = va * vc
        concat_vector = torch.concat([va, vb, vc, vd, diff_ac, prod_ac], dim=1)

        if save_concat_vector is not None:
            filenames = os.listdir(save_concat_vector)
            if len(filenames):
                max_num = max([int(filename.split('.pth')[0]) for filename in filenames])
                torch.save(concat_vector, save_concat_vector + "/{}.pth".format(max_num + 1))
            else:
                torch.save(concat_vector, save_concat_vector + "/1.pth")

        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)

        concat_vector = torch.concat([concat_vector, strainPassCats_vector], dim=1)
        concat_vector = self.hidden_layer(concat_vector)
        concat_vector = self.hidden_act(concat_vector)

        logits = self.classifier(concat_vector)
        output = logits

        outputs = [logits, output]
        if labels is not None:
            if self.output_mode in ["regression"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss] + outputs
        return outputs

class fluProfiler_v1_2_1(PreTrainedModel):
    """fluProfiler model v1.2

    Design goal:
      - Down-project token embeddings (2560 -> 1280)
      - Use existing value-attention mechanism to compute token weights (no pooling)
      - Map to a new interaction space and perform cross-attention (HA: a<->c, NA: b<->d)
      - Pool after cross-attention and regress
      - Keep Passage_encoder unchanged
      - Passage_encoder: dim = 256 -> 16
    """

    config_class = fluProfiler_Config

    def __init__(self, config, args):
        super().__init__(config)

        if config.seq_max_length is None and config.seq_max_length_a == config.seq_max_length_b == config.seq_max_length_c == config.seq_max_length_d:
            config.seq_max_length = config.seq_max_length_a
        if config.matrix_max_length is None and config.matrix_max_length_a == config.matrix_max_length_b == config.matrix_max_length_c == config.matrix_max_length_d:
            config.matrix_max_length = config.matrix_max_length_a
        if config.embedding_input_size is None and config.embedding_input_size_a == config.embedding_input_size_b == config.embedding_input_size_c == config.embedding_input_size_d:
            config.embedding_input_size = config.embedding_input_size_a

        self.num_labels = config.num_labels
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type

        # ---- hyperparams (with safe defaults) ----
        self.in_dim = int(getattr(config, "hidden_size", 2560))
        self.down_dim = int(getattr(config, "down_dim", 1280))
        self.cross_dim = int(getattr(config, "cross_dim", 256))
        self.cross_heads = int(getattr(config, "cross_heads", 4))
        self.dropout_prob = float(getattr(config, "dropout_prob", 0.1))

        if self.cross_dim % self.cross_heads != 0:
            raise ValueError(f"cross_dim ({self.cross_dim}) must be divisible by cross_heads ({self.cross_heads}).")

        # ---- stage 1: down-projection ----
        self.matrix_dropout = nn.Dropout(p=self.dropout_prob)
        self.down_proj = nn.Sequential(
            nn.Linear(self.in_dim, self.down_dim, bias=False),
            nn.LayerNorm(self.down_dim),
        )

        # ---- stage 2: token weighting (value-attention, no pooling) ----
        self.token_weighting = value_weighting(embed_size=self.down_dim)

        # ---- stage 3: map to cross-attention space ----
        self.to_cross = nn.Sequential(
            nn.Linear(self.down_dim, self.cross_dim, bias=False),
            nn.LayerNorm(self.cross_dim),
        )

        # ---- stage 4: cross-attention (single-layer, shared weights) ----
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.cross_dim,
            num_heads=self.cross_heads,
            dropout=self.dropout_prob,
            batch_first=True,
        )
        self.cross_ln = nn.LayerNorm(self.cross_dim)
        self.cross_drop = nn.Dropout(self.dropout_prob)

        # ---- stage 5: pooling after cross-attention ----
        self.post_pool = attention_pooling(embed_size=self.cross_dim)

        # ---- passage encoder (unchanged) ----
        self.Passage_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=6, embedding_dim=16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )

        # ---- head ----
        # We keep the same final feature construction as v0_1:
        # [a, b, c, d, (a-c), (a*c)] then concat passage(256)
        last_hidden_size = self.cross_dim
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = create_loss_function(
            config,
            args,
            hidden_size=last_hidden_size * 6 + 16,
            classifier_size=args.classifier_size,
            sigmoid=args.sigmoid,
            output_mode=args.output_mode,
            num_labels=self.num_labels,
            loss_type=args.loss_type,
            ignore_index=args.ignore_index,
            return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"],
        )

    @staticmethod
    def _to_key_padding_mask(mask):
        """Convert (B,L) float/int mask with 1=valid,0=pad to key_padding_mask bool with True=pad."""
        if mask is None:
            return None
        return mask == 0

    def _encode_side(self, matrices, matrix_attention_masks):
        """Down-project and compute token-wise weighting."""
        x = self.matrix_dropout(matrices)
        x = self.down_proj(x)  # (B,L,down_dim)
        x = self.token_weighting(x, mask=matrix_attention_masks)  # (B,L,down_dim)
        x = self.to_cross(x)  # (B,L,cross_dim)
        return x

    def _cross(self, q, kv, kv_mask):
        """Cross-attention with residual + layer norm."""
        key_padding_mask = self._to_key_padding_mask(kv_mask)
        out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        out = self.cross_ln(q + self.cross_drop(out))
        return out

    def _pool(self, x, mask):
        return self.post_pool(x, mask=mask)

    def forward(
        self,
        input_ids_a=None, input_ids_b=None, input_ids_c=None, input_ids_d=None,
        position_ids_a=None, position_ids_b=None, position_ids_c=None, position_ids_d=None,
        token_type_ids_a=None, token_type_ids_b=None, token_type_ids_c=None, token_type_ids_d=None,
        seq_attention_masks_a=None, seq_attention_masks_b=None, seq_attention_masks_c=None, seq_attention_masks_d=None,
        vectors_a=None, vectors_b=None, vectors_c=None, vectors_d=None,
        matrices_a=None, matrices_b=None, matrices_c=None, matrices_d=None,
        matrix_attention_masks_a=None, matrix_attention_masks_b=None, matrix_attention_masks_c=None, matrix_attention_masks_d=None,
        strainPassCats=None, labels=None,
        save_concat_vector=None,
        save_attention_path=None,
        **kwargs,
    ):
        Za = self._encode_side(matrices_a, matrix_attention_masks_a)
        Zb = self._encode_side(matrices_b, matrix_attention_masks_b)
        Zc = self._encode_side(matrices_c, matrix_attention_masks_c)
        Zd = self._encode_side(matrices_d, matrix_attention_masks_d)

        # Cross-attention in interaction space
        Zac = self._cross(Za, Zc, matrix_attention_masks_c)  # a attends c
        Zca = self._cross(Zc, Za, matrix_attention_masks_a)  # c attends a
        Zbd = self._cross(Zb, Zd, matrix_attention_masks_d)  # b attends d
        Zdb = self._cross(Zd, Zb, matrix_attention_masks_b)  # d attends b

        # Pool after cross-attention
        va = self._pool(Zac, matrix_attention_masks_a)  # (B,cross_dim)
        vb = self._pool(Zbd, matrix_attention_masks_b)
        vc = self._pool(Zca, matrix_attention_masks_c)
        vd = self._pool(Zdb, matrix_attention_masks_d)

        strainPassCats_vector = torch.mean(self.Passage_encoder(strainPassCats), dim=1)

        diff_ac = va - vc
        prod_ac = va * vc
        concat_vector = torch.concat([va, vb, vc, vd, diff_ac, prod_ac], dim=1)

        if save_concat_vector is not None:
            filenames = os.listdir(save_concat_vector)
            if len(filenames):
                max_num = max([int(filename.split('.pth')[0]) for filename in filenames])
                torch.save(concat_vector, save_concat_vector + "/{}.pth".format(max_num + 1))
            else:
                torch.save(concat_vector, save_concat_vector + "/1.pth")

        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)

        concat_vector = torch.concat([concat_vector, strainPassCats_vector], dim=1)
        concat_vector = self.hidden_layer(concat_vector)
        concat_vector = self.hidden_act(concat_vector)

        logits = self.classifier(concat_vector)
        output = logits

        outputs = [logits, output]
        if labels is not None:
            if self.output_mode in ["regression"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss] + outputs
        return outputs

class fluProfiler_v1_2_2(PreTrainedModel):
    """fluProfiler model v1.2.2

    Design goal:
      - Down-project token embeddings (2560 -> 1280)
      - Use existing value-attention mechanism to compute token weights (no pooling)
      - Map to a new interaction space and perform cross-attention (HA: a<->c, NA: b<->d)
      - Pool after cross-attention and regress
      - Keep Passage_encoder unchanged
      - Passage_encoder: dim = 256 -> 16
      - Concat Passage embedding without max pooling
      - Concat Passage embedding with first token
    """

    config_class = fluProfiler_Config

    def __init__(self, config, args):
        super().__init__(config)

        if config.seq_max_length is None and config.seq_max_length_a == config.seq_max_length_b == config.seq_max_length_c == config.seq_max_length_d:
            config.seq_max_length = config.seq_max_length_a
        if config.matrix_max_length is None and config.matrix_max_length_a == config.matrix_max_length_b == config.matrix_max_length_c == config.matrix_max_length_d:
            config.matrix_max_length = config.matrix_max_length_a
        if config.embedding_input_size is None and config.embedding_input_size_a == config.embedding_input_size_b == config.embedding_input_size_c == config.embedding_input_size_d:
            config.embedding_input_size = config.embedding_input_size_a

        self.num_labels = config.num_labels
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type

        # ---- hyperparams (with safe defaults) ----
        self.in_dim = int(getattr(config, "hidden_size", 2560))
        self.down_dim = int(getattr(config, "down_dim", 1280))
        self.cross_dim = int(getattr(config, "cross_dim", 256))
        self.cross_heads = int(getattr(config, "cross_heads", 4))
        self.dropout_prob = float(getattr(config, "dropout_prob", 0.1))

        if self.cross_dim % self.cross_heads != 0:
            raise ValueError(f"cross_dim ({self.cross_dim}) must be divisible by cross_heads ({self.cross_heads}).")

        # ---- stage 1: down-projection ----
        self.matrix_dropout = nn.Dropout(p=self.dropout_prob)
        self.down_proj = nn.Sequential(
            nn.Linear(self.in_dim, self.down_dim, bias=False),
            nn.LayerNorm(self.down_dim),
        )

        # ---- stage 2: token weighting (value-attention, no pooling) ----
        self.token_weighting = value_weighting(embed_size=self.down_dim)

        # ---- stage 3: map to cross-attention space ----
        self.to_cross = nn.Sequential(
            nn.Linear(self.down_dim, self.cross_dim, bias=False),
            nn.LayerNorm(self.cross_dim),
        )

        # ---- stage 4: cross-attention (single-layer, shared weights) ----
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.cross_dim,
            num_heads=self.cross_heads,
            dropout=self.dropout_prob,
            batch_first=True,
        )
        self.cross_ln = nn.LayerNorm(self.cross_dim)
        self.cross_drop = nn.Dropout(self.dropout_prob)

        # ---- stage 5: pooling after cross-attention ----
        self.post_pool = attention_pooling(embed_size=self.cross_dim)

        # ---- passage encoder (unchanged) ----
        self.Passage_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=6, embedding_dim=16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )

        # ---- head ----
        # We keep the same final feature construction as v0_1:
        # [a, b, c, d, (a-c), (a*c)] then concat passage(256)
        last_hidden_size = self.cross_dim
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = create_loss_function(
            config,
            args,
            hidden_size=last_hidden_size * 6 + 32,
            classifier_size=args.classifier_size,
            sigmoid=args.sigmoid,
            output_mode=args.output_mode,
            num_labels=self.num_labels,
            loss_type=args.loss_type,
            ignore_index=args.ignore_index,
            return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"],
        )

    @staticmethod
    def _to_key_padding_mask(mask):
        """Convert (B,L) float/int mask with 1=valid,0=pad to key_padding_mask bool with True=pad."""
        if mask is None:
            return None
        return mask == 0

    def _encode_side(self, matrices, matrix_attention_masks):
        """Down-project and compute token-wise weighting."""
        x = self.matrix_dropout(matrices)
        x = self.down_proj(x)  # (B,L,down_dim)
        x = self.token_weighting(x, mask=matrix_attention_masks)  # (B,L,down_dim)
        x = self.to_cross(x)  # (B,L,cross_dim)
        return x

    def _cross(self, q, kv, kv_mask):
        """Cross-attention with residual + layer norm."""
        key_padding_mask = self._to_key_padding_mask(kv_mask)
        out, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        out = self.cross_ln(q + self.cross_drop(out))
        return out

    def _pool(self, x, mask):
        return self.post_pool(x, mask=mask)

    def forward(
        self,
        input_ids_a=None, input_ids_b=None, input_ids_c=None, input_ids_d=None,
        position_ids_a=None, position_ids_b=None, position_ids_c=None, position_ids_d=None,
        token_type_ids_a=None, token_type_ids_b=None, token_type_ids_c=None, token_type_ids_d=None,
        seq_attention_masks_a=None, seq_attention_masks_b=None, seq_attention_masks_c=None, seq_attention_masks_d=None,
        vectors_a=None, vectors_b=None, vectors_c=None, vectors_d=None,
        matrices_a=None, matrices_b=None, matrices_c=None, matrices_d=None,
        matrix_attention_masks_a=None, matrix_attention_masks_b=None, matrix_attention_masks_c=None, matrix_attention_masks_d=None,
        strainPassCats=None, labels=None,
        save_concat_vector=None,
        save_attention_path=None,
        **kwargs,
    ):
        Za = self._encode_side(matrices_a, matrix_attention_masks_a)
        Zb = self._encode_side(matrices_b, matrix_attention_masks_b)
        Zc = self._encode_side(matrices_c, matrix_attention_masks_c)
        Zd = self._encode_side(matrices_d, matrix_attention_masks_d)

        # Cross-attention in interaction space
        Zac = self._cross(Za, Zc, matrix_attention_masks_c)  # a attends c
        Zca = self._cross(Zc, Za, matrix_attention_masks_a)  # c attends a
        Zbd = self._cross(Zb, Zd, matrix_attention_masks_d)  # b attends d
        Zdb = self._cross(Zd, Zb, matrix_attention_masks_b)  # d attends b

        # Pool after cross-attention
        va = self._pool(Zac, matrix_attention_masks_a)  # (B,cross_dim)
        vb = self._pool(Zbd, matrix_attention_masks_b)
        vc = self._pool(Zca, matrix_attention_masks_c)
        vd = self._pool(Zdb, matrix_attention_masks_d)

        strainPassCats_vector = self.Passage_encoder(strainPassCats).reshape(strainPassCats.size(0), -1)

        diff_ac = va - vc
        prod_ac = va * vc
        concat_vector = torch.concat([va, vb, vc, vd, diff_ac, prod_ac], dim=1)

        if save_concat_vector is not None:
            filenames = os.listdir(save_concat_vector)
            if len(filenames):
                max_num = max([int(filename.split('.pth')[0]) for filename in filenames])
                torch.save(concat_vector, save_concat_vector + "/{}.pth".format(max_num + 1))
            else:
                torch.save(concat_vector, save_concat_vector + "/1.pth")

        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)

        concat_vector = torch.concat([concat_vector, strainPassCats_vector], dim=1)
        concat_vector = self.hidden_layer(concat_vector)
        concat_vector = self.hidden_act(concat_vector)

        logits = self.classifier(concat_vector)
        output = logits

        outputs = [logits, output]
        if labels is not None:
            if self.output_mode in ["regression"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                loss = self.loss_fct(logits, labels.float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss] + outputs
        return outputs


class fluProfiler(PreTrainedModel):
    """
    HANA baseline in current fluProfiler framework.

    Uses only matrix embeddings (token-level embeddings) and an attention_mask pooler,
    then concatenates a/b/c/d, adds passage embedding, and does regression/classification
    via create_loss_function.
    """

    config_class = fluProfiler_Config

    def __init__(self, config, args):
        super().__init__(config)

        # ---- unify config fields (keep same behavior as your old code) ----
        if config.seq_max_length is None and config.seq_max_length_a == config.seq_max_length_b == config.seq_max_length_c == config.seq_max_length_d:
            config.seq_max_length = config.seq_max_length_a
        if config.matrix_max_length is None and config.matrix_max_length_a == config.matrix_max_length_b == config.matrix_max_length_c == config.matrix_max_length_d:
            config.matrix_max_length = config.matrix_max_length_a
        if config.embedding_input_size is None and config.embedding_input_size_a == config.embedding_input_size_b == config.embedding_input_size_c == config.embedding_input_size_d:
            config.embedding_input_size = config.embedding_input_size_a

        self.num_labels = config.num_labels
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type

        self.fusion_type = getattr(args, "fusion_type", None) or "concat"
        self.prepend_bos = getattr(args, "prepend_bos", False)
        self.append_eos = getattr(args, "append_eos", False)

        self.matrix_dropout = nn.Dropout(p=0.1)

        # ---- matrix pooler (same as old code) ----
        # old: new_config.embedding_input_size = config.hidden_size; attention_mask(embed_size=hidden_size)
        self.hidden_size = config.hidden_size
        self.matrix_pooler = attention_mask(embed_size=self.hidden_size)

        # ---- optional matrix FC stack (compatible with config.matrix_fc_size) ----
        # config.matrix_fc_size can be: None, "512", or [512, 256] etc.
        self.matrix_fc = None
        matrix_fc_size = getattr(config, "matrix_fc_size", None)
        out_dim = self.hidden_size

        if matrix_fc_size is not None and (isinstance(matrix_fc_size, list) and len(matrix_fc_size) > 0 or isinstance(matrix_fc_size, (str, int))):
            if not isinstance(matrix_fc_size, list):
                matrix_fc_size = [int(matrix_fc_size)]
            else:
                matrix_fc_size = [int(v) for v in matrix_fc_size]

            layers = []
            in_dim = self.hidden_size
            for dim in matrix_fc_size:
                layers.append(nn.Linear(in_dim, dim))
                in_dim = dim
            self.matrix_fc = nn.Sequential(*layers)
            out_dim = matrix_fc_size[-1]

        self.single_repr_dim = out_dim  # representation dim for each of a/b/c/d after pooling(+fc)

        # ---- head (same structure as old code) ----
        # concat a,b,c,d -> 4 * single_repr_dim
        # then concat passage vector -> +256
        head_in = self.single_repr_dim * 4 + 256

        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = create_loss_function(
            config,
            args,
            hidden_size=head_in,
            classifier_size=args.classifier_size,
            sigmoid=args.sigmoid,
            output_mode=args.output_mode,
            num_labels=self.num_labels,
            loss_type=args.loss_type,
            ignore_index=args.ignore_index,
            return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"],
        )

        # ---- passage encoder (kept as-is from your old code) ----
        # 注意：你旧代码是 num_embeddings=5，但你项目里很多地方用 6（含 special/pad）
        # 我这里建议保守用 6，避免越界。
        self.Passage_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=6, embedding_dim=256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def _encode_matrix(self, matrices, matrix_attention_masks, save_attention_path=None):
        """
        matrices: (B, L, hidden_size)
        mask: (B, L) with 1 for valid tokens, 0 for pad
        """
        if matrices is None:
            raise ValueError("matrices is None but fluProfiler_HANA expects matrices input.")

        x = self.matrix_dropout(matrices)
        vec = self.matrix_pooler(x, mask=matrix_attention_masks, save_attention_path=save_attention_path)  # (B, hidden_size) or (B, out_dim)
        if self.matrix_fc is not None:
            vec = self.matrix_fc(vec)
        return vec

    def forward(
        self,
        input_ids_a=None, input_ids_b=None, input_ids_c=None, input_ids_d=None,
        position_ids_a=None, position_ids_b=None, position_ids_c=None, position_ids_d=None,
        token_type_ids_a=None, token_type_ids_b=None, token_type_ids_c=None, token_type_ids_d=None,
        seq_attention_masks_a=None, seq_attention_masks_b=None, seq_attention_masks_c=None, seq_attention_masks_d=None,
        vectors_a=None, vectors_b=None, vectors_c=None, vectors_d=None,
        matrices_a=None, matrices_b=None, matrices_c=None, matrices_d=None,
        matrix_attention_masks_a=None, matrix_attention_masks_b=None, matrix_attention_masks_c=None, matrix_attention_masks_d=None,
        strainPassCats=None, labels=None,
        save_concat_vector=None,
        save_attention_path=None,
        **kwargs
    ):
        # Encode each matrix input
        rep_a = self._encode_matrix(matrices_a, matrix_attention_masks_a, save_attention_path=save_attention_path)
        rep_b = self._encode_matrix(matrices_b, matrix_attention_masks_b, save_attention_path=save_attention_path)
        rep_c = self._encode_matrix(matrices_c, matrix_attention_masks_c, save_attention_path=save_attention_path)
        rep_d = self._encode_matrix(matrices_d, matrix_attention_masks_d, save_attention_path=save_attention_path)

        # Passage vector (kept same behavior: mean over token dimension)
        if strainPassCats is None:
            raise ValueError("strainPassCats is required for fluProfiler_HANA (passage encoder).")
        passage_vec = torch.mean(self.Passage_encoder(strainPassCats), dim=1)  # (B,256)

        # Concat a/b/c/d
        concat_vector = torch.cat([rep_a, rep_b, rep_c, rep_d], dim=1)  # (B, 4*single_repr_dim)

        # Optional save
        if save_concat_vector is not None:
            os.makedirs(save_concat_vector, exist_ok=True)
            filenames = os.listdir(save_concat_vector)
            if len(filenames):
                max_num = max([int(fn.split('.pth')[0]) for fn in filenames if fn.endswith(".pth")])
                torch.save(concat_vector, os.path.join(save_concat_vector, f"{max_num + 1}.pth"))
            else:
                torch.save(concat_vector, os.path.join(save_concat_vector, "1.pth"))

        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)

        # Add passage features
        concat_vector = torch.cat([concat_vector, passage_vec], dim=1)  # (B, 4*single_repr_dim + 256)

        # Head
        h = self.hidden_layer(concat_vector)
        h = self.hidden_act(h)

        logits = self.classifier(h)
        output = logits

        outputs = [logits, output]

        # Loss (keep same general logic as your other architectures)
        if labels is not None:
            if self.output_mode in ["regression"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss] + outputs

        return outputs


class AdaptivePooling(nn.Module):
    """
    自适应池化：结合注意力池化和值池化
    """
    def __init__(self, embed_dim):
        super(AdaptivePooling, self).__init__()
        self.attention_pool = attention_pooling(embed_dim)
        self.value_pool = value_pooling(embed_dim)
        self.gate = nn.Linear(embed_dim * 2, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None, save_attention_path=None):
        # 分别进行两种池化
        attn_out = self.attention_pool(x, mask=mask, save_attention_path=save_attention_path)
        val_out = self.value_pool(x, mask=mask)

        # 学习如何组合两种池化结果
        combined = torch.cat([attn_out, val_out], dim=-1)  # (B, 2*embed_dim)
        weights = F.softmax(self.gate(combined), dim=-1)   # (B, 2)

        # 加权组合
        output = weights[:, 0:1] * attn_out + weights[:, 1:2] * val_out
        return self.dropout(output)


class CrossAttentionFusion(nn.Module):
    """
    跨句子注意力融合机制
    """
    def __init__(self, embed_dim, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, representations):
        """
        Args:
            representations: List of tensors [rep_a, rep_b, rep_c, rep_d], each (B, embed_dim)
        Returns:
            fused_representations: (B, embed_dim)
        """
        # Stack representations: (4, B, embed_dim) -> (B, 4, embed_dim)
        stacked = torch.stack(representations, dim=1)  # (B, 4, embed_dim)

        # Self-attention among the 4 representations
        B, N, D = stacked.shape

        # Linear projections
        q = self.q_proj(stacked).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)
        k = self.k_proj(stacked).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)
        v = self.v_proj(stacked).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, head_dim)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, N, N)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (B, H, N, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)  # (B, N, D)

        # Output projection and residual connection
        output = self.out_proj(attn_output)
        output = self.layer_norm(output + stacked)

        # Pool across the sequence dimension (mean pooling for now)
        fused = output.mean(dim=1)  # (B, D)

        return fused


class EnhancedPassageEncoder(nn.Module):
    """
    增强的Passage Encoder，包含残差连接和更深层结构
    """
    def __init__(self, vocab_size=6, embed_dim=256, hidden_dim=512):
        super(EnhancedPassageEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim)
            ) for _ in range(3)  # 3层
        ])
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len) - passage token ids
        Returns:
            output: (B, embed_dim)
        """
        x = self.embedding(x)  # (B, seq_len, embed_dim)

        # Apply transformer-like layers with residual connections
        for layer in self.layers:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection

        # Global average pooling across sequence dimension
        x = x.mean(dim=1)  # (B, embed_dim)
        x = self.dropout(x)

        return x


class fluProfiler_v1_0(PreTrainedModel):
    """
    改进版 fluProfiler v1.0

    主要改进：
    1. 使用自适应池化结合注意力池化和值池化
    2. 跨句子注意力机制用于特征融合
    3. 增强的Passage Encoder
    4. 更深的网络结构和残差连接
    """
    config_class = fluProfiler_Config

    def __init__(self, config, args):
        super(fluProfiler_v1_0, self).__init__(config)
        if config.seq_max_length is None and config.seq_max_length_a == config.seq_max_length_b == config.seq_max_length_c == config.seq_max_length_d:
            config.seq_max_length = config.seq_max_length_a
        if config.matrix_max_length is None and config.matrix_max_length_a == config.matrix_max_length_b == config.matrix_max_length_c == config.matrix_max_length_d:
            config.matrix_max_length = config.matrix_max_length_a
        if config.embedding_input_size is None and config.embedding_input_size_a == config.embedding_input_size_b == config.embedding_input_size_c == config.embedding_input_size_d:
            config.embedding_input_size = config.embedding_input_size_a

        self.num_labels = config.num_labels
        self.fusion_type = args.fusion_type if hasattr(args, "fusion_type") and args.fusion_type else "attention"
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type
        self.prepend_bos = args.prepend_bos
        self.append_eos = args.append_eos

        # 使用自适应池化
        new_config = copy.deepcopy(config)
        new_config.embedding_input_size = config.hidden_size
        self.matrix_pooler = AdaptivePooling(embed_size=new_config.embedding_input_size)

        # 跨句子注意力融合
        self.cross_attention_fusion = CrossAttentionFusion(embed_dim=config.hidden_size)

        # 增强的Passage Encoder
        self.Passage_encoder = EnhancedPassageEncoder(vocab_size=6, embed_dim=256, hidden_dim=512)

        # 创建损失函数
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(
                config,
                args,
                hidden_size=config.hidden_size * 2,  # 融合后的特征 + passage特征
                classifier_size=args.classifier_size,
                sigmoid=args.sigmoid,
                output_mode=args.output_mode,
                num_labels=self.num_labels,
                loss_type=args.loss_type,
                ignore_index=args.ignore_index,
                return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
            )

    def __forward__(
            self,
            matrices,
            matrix_attention_masks,
            save_attention_path
    ):
        if matrices is not None:
            matrices = self.matrix_pooler(matrices, mask=matrix_attention_masks, save_attention_path=save_attention_path)

        return matrices

    def forward(
            self,
            input_ids_a=None, input_ids_b=None, input_ids_c=None, input_ids_d=None,
            position_ids_a=None, position_ids_b=None, position_ids_c=None, position_ids_d=None,
            token_type_ids_a=None, token_type_ids_b=None, token_type_ids_c=None, token_type_ids_d=None,
            seq_attention_masks_a=None, seq_attention_masks_b=None, seq_attention_masks_c=None, seq_attention_masks_d=None,
            vectors_a=None, vectors_b=None, vectors_c=None, vectors_d=None,
            matrices_a=None, matrices_b=None, matrices_c=None, matrices_d=None,
            matrix_attention_masks_a=None, matrix_attention_masks_b=None, matrix_attention_masks_c=None, matrix_attention_masks_d=None,
            strainPassCats=None, labels=None,
            save_concat_vector=None,
            save_attention_path=None,
            **kwargs
    ):
        # 获取四个序列的表示
        representation_vector_a = self.__forward__(
            matrices_a,
            matrix_attention_masks_a,
            save_attention_path
        )

        representation_vector_b = self.__forward__(
            matrices_b,
            matrix_attention_masks_b,
            save_attention_path
        )

        representation_vector_c = self.__forward__(
            matrices_c,
            matrix_attention_masks_c,
            save_attention_path
        )

        representation_vector_d = self.__forward__(
            matrices_d,
            matrix_attention_masks_d,
            save_attention_path
        )

        # 使用跨句子注意力融合
        representations = [representation_vector_a, representation_vector_b,
                          representation_vector_c, representation_vector_d]
        fused_vector = self.cross_attention_fusion(representations)

        # Passage特征编码
        strainPassCats_vector = self.Passage_encoder(strainPassCats)

        # 最终特征融合
        concat_vector = torch.cat([fused_vector, strainPassCats_vector], dim=1)

        # 前向传播
        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)

        concat_vector = self.hidden_layer(concat_vector)
        concat_vector = self.hidden_act(concat_vector)

        logits = self.classifier(concat_vector)
        output = logits

        outputs = [logits, output]
        if labels is not None:
            if self.output_mode in ["regression"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    loss = self.loss_fct(logits, labels)
                else:
                    loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    loss = self.loss_fct(logits, labels.float())
                else:
                    loss = self.loss_fct(logits.view(-1), labels.view(-1).float())

            outputs = [loss, *outputs]
        return outputs


class LightweightFusion(nn.Module):
    """
    轻量级特征融合模块
    
    参数增加：1个Linear层 + LayerNorm ≈ embed_dim^2 + 2*embed_dim
    """
    def __init__(self, embed_dim):
        super(LightweightFusion, self).__init__()
        # 只添加1个Linear层进行特征变换
        self.fusion_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, rep_a, rep_b, rep_c, rep_d):
        """
        Args:
            rep_a, rep_b, rep_c, rep_d: (B, embed_dim)
        Returns:
            concat_vector: (B, 6*embed_dim)
        """
        # 保留原有的交互方式（无参数）
        diff_ac = rep_a - rep_c
        prod_ac = rep_a * rep_c
        
        # 对每个表示进行轻量级变换
        rep_a = self.layer_norm(self.fusion_proj(rep_a))
        rep_b = self.layer_norm(self.fusion_proj(rep_b))
        rep_c = self.layer_norm(self.fusion_proj(rep_c))
        rep_d = self.layer_norm(self.fusion_proj(rep_d))
        
        # 组合：原始4个 + diff + prod = 6个
        concat = torch.cat([rep_a, rep_b, rep_c, rep_d, diff_ac, prod_ac], dim=1)
        return self.dropout(concat)


class ImprovedPassageEncoder(nn.Module):
    """
    改进的Passage Encoder
    
    参数增加：2层网络（参数共享）≈ embed_dim^2 + 2*embed_dim
    """
    def __init__(self, vocab_size=6, embed_dim=256):
        super(ImprovedPassageEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 2层网络，参数共享（同一层用于所有位置）
        self.layer1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, seq_len) - passage token ids
        Returns:
            output: (B, embed_dim)
        """
        x = self.embedding(x)  # (B, seq_len, embed_dim)
        
        # 参数共享：同一层用于所有位置
        x = self.layer1(x)
        x = x + self.layer2(x)  # 残差连接
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, embed_dim)
        return x


class fluProfiler_v1_1(PreTrainedModel):
    """
    均衡优化的 fluProfiler v1.1
    
    主要改进（参数增加<10%）：
    1. 池化部分：添加Layer Normalization（少量参数）
    2. 特征融合：轻量级融合层（1个Linear + LayerNorm）
    3. Passage编码：2层网络（参数共享）
    4. 训练稳定性：各部分添加LayerNorm和正则化
    
    设计原则：
    - 参数增加最小化（<10%）
    - 负载均衡到各个部分
    - 避免过拟合
    """
    config_class = fluProfiler_Config

    def __init__(self, config, args):
        super(fluProfiler_v1_1, self).__init__(config)
        if config.seq_max_length is None and config.seq_max_length_a == config.seq_max_length_b == config.seq_max_length_c == config.seq_max_length_d:
            config.seq_max_length = config.seq_max_length_a
        if config.matrix_max_length is None and config.matrix_max_length_a == config.matrix_max_length_b == config.matrix_max_length_c == config.matrix_max_length_d:
            config.matrix_max_length = config.matrix_max_length_a
        if config.embedding_input_size is None and config.embedding_input_size_a == config.embedding_input_size_b == config.embedding_input_size_c == config.embedding_input_size_d:
            config.embedding_input_size = config.embedding_input_size_a

        self.num_labels = config.num_labels
        self.fusion_type = args.fusion_type if hasattr(args, "fusion_type") and args.fusion_type else "lightweight"
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type
        self.prepend_bos = args.prepend_bos
        self.append_eos = args.append_eos

        # 池化部分：保持value_pooling，添加LayerNorm
        self.matrix_dropout = nn.Dropout(p=0.1)
        new_config = copy.deepcopy(config)
        new_config.embedding_input_size = config.hidden_size
        self.matrix_pooler = value_pooling(embed_size=new_config.embedding_input_size)
        self.pooler_norm = nn.LayerNorm(config.hidden_size)  # 池化后归一化

        # 特征融合部分：轻量级融合层
        self.fusion = LightweightFusion(embed_dim=config.hidden_size)

        # Passage编码部分：改进的编码器
        self.Passage_encoder = ImprovedPassageEncoder(vocab_size=6, embed_dim=256)

        # 分类器部分：添加LayerNorm稳定训练
        fusion_output_size = config.hidden_size * 6  # 融合后的特征维度
        passage_size = 256  # Passage特征维度
        total_hidden_size = fusion_output_size + passage_size
        
        self.pre_classifier_norm = nn.LayerNorm(total_hidden_size)  # 分类器前归一化
        
        # 创建损失函数
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(
                config,
                args,
                hidden_size=total_hidden_size,
                classifier_size=args.classifier_size,
                sigmoid=args.sigmoid,
                output_mode=args.output_mode,
                num_labels=self.num_labels,
                loss_type=args.loss_type,
                ignore_index=args.ignore_index,
                return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
            )

    def __forward__(
            self,
            matrices,
            matrix_attention_masks,
            save_attention_path
    ):
        if matrices is not None:
            matrices = self.matrix_dropout(matrices)
            matrix_vector = self.matrix_pooler(matrices, mask=matrix_attention_masks, save_attention_path=save_attention_path)
            # 池化后归一化（稳定训练）
            matrix_vector = self.pooler_norm(matrix_vector)

        return matrix_vector

    def forward(
            self,
            input_ids_a=None, input_ids_b=None, input_ids_c=None, input_ids_d=None,
            position_ids_a=None, position_ids_b=None, position_ids_c=None, position_ids_d=None,
            token_type_ids_a=None, token_type_ids_b=None, token_type_ids_c=None, token_type_ids_d=None,
            seq_attention_masks_a=None, seq_attention_masks_b=None, seq_attention_masks_c=None, seq_attention_masks_d=None,
            vectors_a=None, vectors_b=None, vectors_c=None, vectors_d=None,
            matrices_a=None, matrices_b=None, matrices_c=None, matrices_d=None,
            matrix_attention_masks_a=None, matrix_attention_masks_b=None, matrix_attention_masks_c=None, matrix_attention_masks_d=None,
            strainPassCats=None, labels=None,
            save_concat_vector=None,
            save_attention_path=None,
            **kwargs
    ):
        # 获取四个序列的表示
        representation_vector_a = self.__forward__(
            matrices_a,
            matrix_attention_masks_a,
            save_attention_path
        )

        representation_vector_b = self.__forward__(
            matrices_b,
            matrix_attention_masks_b,
            save_attention_path
        )

        representation_vector_c = self.__forward__(
            matrices_c,
            matrix_attention_masks_c,
            save_attention_path
        )

        representation_vector_d = self.__forward__(
            matrices_d,
            matrix_attention_masks_d,
            save_attention_path
        )

        # 轻量级特征融合
        concat_vector = self.fusion(
            representation_vector_a,
            representation_vector_b,
            representation_vector_c,
            representation_vector_d
        )

        # Passage特征编码
        strainPassCats_vector = self.Passage_encoder(strainPassCats)

        # 最终特征融合
        concat_vector = torch.cat([concat_vector, strainPassCats_vector], dim=1)

        # 分类器前归一化（稳定训练）
        concat_vector = self.pre_classifier_norm(concat_vector)

        # 前向传播
        if self.dropout is not None:
            concat_vector = self.dropout(concat_vector)

        concat_vector = self.hidden_layer(concat_vector)
        concat_vector = self.hidden_act(concat_vector)

        logits = self.classifier(concat_vector)
        output = logits

        outputs = [logits, output]
        if labels is not None:
            if self.output_mode in ["regression"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    loss = self.loss_fct(logits, labels)
                else:
                    loss = self.loss_fct(logits.view(-1), labels.view(-1))
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    loss = self.loss_fct(logits, labels.float())
                else:
                    loss = self.loss_fct(logits.view(-1), labels.view(-1).float())

            outputs = [loss, *outputs]
        return outputs


# Additional model variants would be added here...
# (fluProfiler, LucaQuadruple_final_dropout, etc.)