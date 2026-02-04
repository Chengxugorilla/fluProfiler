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
from .pooling import value_pooling, attention_mask


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
        # self.matrix_pooler = attention_mask(embed_size=new_config.embedding_input_size)
        # self.matrix_pooler = attention_pooling(embed_size=new_config.embedding_input_size)
        # self.matrix_pooler = ResidueFeaturePooling(embed_size=new_config.embedding_input_size)
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


# Additional model variants would be added here...
# (fluProfiler, LucaQuadruple_final_dropout, etc.)