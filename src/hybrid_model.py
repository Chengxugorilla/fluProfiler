# encoding: utf-8

import sys
import logging
import torch.nn as nn
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")
sys.path.append("../../../src")
try:
    from ....common.pooling import *
    from ....common.loss import *
    from ....utils import *
    from ....common.multi_label_metrics import *
    from ....common.modeling_bert import BertModel, BertPreTrainedModel
    from ....common.metrics import *
except ImportError:
    from src.common.pooling import *
    from src.common.loss import *
    from src.utils import *
    from src.common.multi_label_metrics import *
    from src.common.metrics import *
    from src.common.modeling_bert import BertModel, BertPreTrainedModel
logger = logging.getLogger(__name__)

class LucaQuadruple_final_dropout(BertPreTrainedModel):
    def __init__(self, config, args):
        super(LucaQuadruple_final_dropout, self).__init__(config)
        if config.seq_max_length is None and config.seq_max_length_a == config.seq_max_length_b == config.seq_max_length_c == config.seq_max_length_d:
            config.seq_max_length = config.seq_max_length_a
        if config.matrix_max_length is None and config.matrix_max_length_a == config.matrix_max_length_b == config.matrix_max_length_c == config.matrix_max_length_d:
            config.matrix_max_length = config.matrix_max_length_a
        if config.embedding_input_size is None and config.embedding_input_size_a == config.embedding_input_size_b == config.embedding_input_size_c == config.embedding_input_size_d:
            config.embedding_input_size = config.embedding_input_size_a

        self.input_type = args.input_type
        self.num_labels = config.num_labels
        self.fusion_type = args.fusion_type if hasattr(args, "fusion_type") and args.fusion_type else "concat"
        self.output_mode = args.output_mode
        self.task_level_type = args.task_level_type
        self.prepend_bos = args.prepend_bos
        self.append_eos = args.append_eos

        if self.task_level_type not in ["seq_level"]:
            assert self.input_type not in ["vector", "seq_vector"]
            assert self.fusion_type == "add"

        self.seq_encoder, self.seq_pooler, \
        self.matrix_encoder, self.matrix_pooler = None, None, None, None
        self.encoder_type_list = [False, False, False]
        self.input_size_list = [0, 0, 0]
        self.linear_idx = [-1, -1, -1]

        self.matrix_dropout = nn.Dropout(p=0.1)
        if self.input_type == "matrix":
            # emb matrix -> (encoder) - > (pooler) -> fc * -> classifier
            if args.matrix_encoder:
                matrix_encoder_config = copy.deepcopy(config)
                matrix_encoder_config.no_position_embeddings = True
                matrix_encoder_config.no_token_type_embeddings = True
                matrix_encoder_config.max_position_embeddings = config.matrix_max_length
                if args.matrix_encoder_act:
                    self.matrix_encoder_act = True
                    origin_embedding_input_size = config.embedding_input_size
                    matrix_encoder_config.embedding_input_size_new = config.hidden_size
                    self.matrix_encoder = nn.ModuleList([
                        nn.Linear(origin_embedding_input_size, config.hidden_size), # 这里降维到了hidden_size但是下游的Bert层会有一个维度错误的线性层
                        create_activate(config.emb_activate_func),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        BertModel( 
                            matrix_encoder_config,
                            use_pretrained_embedding=True,
                            add_pooling_layer=(args.matrix_pooling_type is None or args.matrix_pooling_type == "none") and self.task_level_type in ["seq_level"]
                        )
                    ])
                else:
                    self.matrix_encoder = nn.ModuleList([
                        # nn.Linear(config.embedding_input_size, config.hidden_size),
                        # nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                        BertModel(
                            matrix_encoder_config,
                            use_pretrained_embedding=True,
                            add_pooling_layer=(args.matrix_pooling_type is None or args.matrix_pooling_type == "none") and self.task_level_type in ["seq_level"]
                        )
                    ])
                ori_embedding_input_size = config.embedding_input_size
                config.embedding_input_size = config.hidden_size
                if self.task_level_type in ["seq_level"]:
                    self.matrix_pooler = create_pooler(pooler_type="matrix", config=config, args=args)
                self.input_size_list[1] = config.embedding_input_size
                config.embedding_input_size = ori_embedding_input_size
            else:
                self.input_size_list[1] = config.hidden_size
                if self.task_level_type in ["seq_level"]:
                    new_config = copy.deepcopy(config)
                    new_config.embedding_input_size = config.hidden_size
                    self.matrix_pooler = create_pooler(pooler_type="matrix", config=new_config, args=args)
            self.encoder_type_list[1] = True
            self.linear_idx[1] = 0
        else:
            raise Exception("Not support input_type=%s" % self.input_type)
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
                    linear_list.append(create_activate(config.fc_activate_func))
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
        if self.fusion_type == "add":
            output_size = [v for v in self.output_size if v > 0]
            assert len(set(output_size)) == 1
            last_hidden_size = output_size[0]
        else:
            last_hidden_size = sum(self.output_size)
        last_hidden_size = last_hidden_size
        self.dropout, self.hidden_layer, self.hidden_act, self.classifier, self.output, self.loss_fct = \
            create_loss_function(
                config,
                args,
                hidden_size=last_hidden_size * 4 + 256,
                classifier_size=args.classifier_size,
                sigmoid=args.sigmoid,
                output_mode=args.output_mode,
                num_labels=self.num_labels,
                loss_type=args.loss_type,
                ignore_index=args.ignore_index,
                return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
            )
        
        self.Passage_encoder = nn.Sequential(
            nn.Embedding(num_embeddings=5, embedding_dim=256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

    def __forward__(
            self,
            input_ids,
            seq_attention_masks,
            token_type_ids,
            position_ids,
            vectors,
            matrices,
            matrix_attention_masks
    ):  
        if matrices is not None:
            matrices = self.matrix_dropout(matrices)
            if self.matrix_encoder is not None:
                if self.matrix_encoder_act is not None:
                    matrices = self.matrix_encoder[0](matrices)
                    for module in self.matrix_encoder[1:-1]:
                        matrices = module(matrices)
                else:
                    print(1/0)
                matrices_output = self.matrix_encoder[-1](
                    input_ids=None,
                    attention_mask=matrix_attention_masks,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=matrices,
                    output_attentions=None,
                    output_hidden_states=None,
                    return_dict=False
                )
                matrices = matrices_output[0]
            
            if self.matrix_pooler is not None: #这里应该是attention pooling
                matrix_vector = self.matrix_pooler(matrices, mask=matrix_attention_masks)
            elif self.task_level_type in ["seq_level"]:
                tmp_mask = torch.unsqueeze(matrix_attention_masks, dim=-1)
                matrices = matrices.masked_fill(tmp_mask == 0, 0.0)
                # 均值pooling
                matrix_vector = torch.sum(matrices, dim=1)/(torch.sum(tmp_mask, dim=1) + 1e-12)
            else:
                matrix_vector = matrices
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
            **kwargs
    ):  
        representation_vector_a = self.__forward__(
            input_ids_a,
            seq_attention_masks_a,
            token_type_ids_a,
            position_ids_a,
            vectors_a,
            matrices_a,
            matrix_attention_masks_a
        )

        representation_vector_b = self.__forward__(
            input_ids_b,
            seq_attention_masks_b,
            token_type_ids_b,
            position_ids_b,
            vectors_b,
            matrices_b,
            matrix_attention_masks_b
        )

        representation_vector_c = self.__forward__(
            input_ids_c,
            seq_attention_masks_c,
            token_type_ids_c,
            position_ids_c,
            vectors_c,
            matrices_c,
            matrix_attention_masks_c
        )

        representation_vector_d = self.__forward__(
            input_ids_d,
            seq_attention_masks_d,
            token_type_ids_d,
            position_ids_d,
            vectors_d,
            matrices_d,
            matrix_attention_masks_d
        )

        strainPassCats_vector = torch.mean(self.Passage_encoder(strainPassCats), dim=1)
        
        concat_vector = torch.concat([representation_vector_a, representation_vector_b, representation_vector_c, representation_vector_d], dim=1)
        
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

        if self.hidden_layer is not None:
            concat_vector = self.hidden_layer(concat_vector)
        if self.hidden_act is not None:
            concat_vector = self.hidden_act(concat_vector)

        logits = self.classifier(concat_vector)
        if self.output:
            output = self.output(logits)
        else:
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
            elif self.output_mode in ["multi_label", "multi-label"]:
                if self.loss_reduction == "meanmean":
                    # logits: N , label_size
                    # labels: N , label_size
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # logits: N , label_size
                    # labels: N , label_size
                    loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels).float())
            elif self.num_labels <= 2 or self.output_mode in ["binary_class", "binary-class"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N ,seq_len, 1
                    # labels: N, seq_len
                    loss = self.loss_fct(logits, labels.float())
                else:
                    # logits: N * seq_len * 1
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            elif self.output_mode in ["multi_class", "multi-class"]:
                if self.task_level_type not in ["seq_level"] and self.loss_reduction == "meanmean":
                    # logits: N ,seq_len, label_size
                    # labels: N , seq_len
                    loss = self.loss_fct(logits, labels)
                else:
                    # logits: N * seq_len, label_size
                    # labels: N * seq_len
                    loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, *outputs]
        return outputs