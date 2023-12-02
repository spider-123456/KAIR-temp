
#KAIR
################################################


"""
import math
import random
import numpy as np

import torch, gc
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import TransformerEncoder

# 新增
import torch.nn.functional as F
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
# gc.collect()
# torch.cuda.empty_cache()
class KAIR(SequentialRecommender):
   
    def __init__(self, config, dataset):
        super(KAIR, self).__init__(config, dataset)

        self.train_stage = config["train_stage"]  # pretrain or finetune


        # load dataset info
        self.ENTITY_ID = config["ENTITY_ID_FIELD"]
        self.RELATION_ID = config["RELATION_ID_FIELD"]
        self.n_entities = dataset.num(self.ENTITY_ID)
        self.n_relations = dataset.num(self.RELATION_ID) - 1
        self.entity_embedding_matrix = dataset.get_preload_weight("ent_id")
        self.relation_embedding_matrix = dataset.get_preload_weight("rel_id")


        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.embedding_size = config["embedding_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.freeze_kg = config["freeze_kg"]
        self.gamma = config["gamma"]  # Scaling factor

        #DA
        self.ssl = config['contrast']
        self.tau = config['tau']
        self.sim = config['sim']
        self.batch_size = config['train_batch_size']
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.sem_aug_nce_fct = nn.CrossEntropyLoss()
        self.lmd=config['lmd']


        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.kg_embedding_size = config["hidden_size"]

        #interest
        self.n_factors = config["n_factors"]
        self.ind =  config["ind"]
        self.sim_decay = config["sim_regularity"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )



        self.loss_fct = nn.CrossEntropyLoss()

        # modules for finetune
        if self.loss_type == "BPR" and self.train_stage == "finetune":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE" and self.train_stage == "finetune":
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.train_stage == "finetune":
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")



        self.pre_model_path = config["pre_model_path"]
        # parameters initialization
        assert self.train_stage in ["pretrain", "finetune"]
        if self.train_stage == "pretrain":
            self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
            self.nce_fct = nn.CrossEntropyLoss()
            self.apply(self._init_weights)
            # 新增
            self.nce_loss = InfoNCELoss(temperature=self.tau,
                                        similarity_type=self.sim)
        else:
            # load pretrained model for finetune
            pretrained = torch.load(self.pre_model_path)
            self.logger.info(f"Load pretrained model from {self.pre_model_path}")
            self.load_state_dict(pretrained["state_dict"],False)
            disen_weight_att = nn.init.xavier_uniform_(torch.empty(self.n_factors, self.n_relations))
            self.disen_weight_att = nn.Parameter(disen_weight_att)
                   
            self.entity_embedding = nn.Embedding( self.n_items, self.kg_embedding_size, padding_idx=0 )
            self.entity_embedding.weight.requires_grad = not self.freeze_kg


            self.dense = nn.Linear(self.hidden_size, self.kg_embedding_size)
            self.dense_layer_u = nn.Linear( self.embedding_size * 2, self.embedding_size )
            self.dense_layer_i = nn.Linear( self.embedding_size * 2, self.embedding_size )

            self.entity_embedding.weight.data.copy_(torch.from_numpy(self.entity_embedding_matrix[: self.n_items]))
            self.relation_Matrix = torch.from_numpy(self.relation_embedding_matrix[: self.n_relations]).to(self.device)  # [R K]
        # 新增
        self.k_intention = config.k_intention
        self.disentangle_encoder = DisentangleEncoder(k_intention=self.k_intention,
                                                      embed_size=self.embedding_size,
                                                      max_len=self.max_seq_length)
        self.score_type = config["score_type"]



    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization

            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_kg_embedding(self, head, disen_weight):
        """Difference:
        We generate the embeddings of the tail entities on every relations only for head due to the 1-N problems.
        """

        head_e = self.entity_embedding(head)  # [B H]
        relation_Matrix = disen_weight.repeat(head_e.size()[0], 1, 1)  # [B R H]
        head_Matrix = torch.unsqueeze(head_e, 1).repeat(
            1, self.n_factors, 1
        )  # [B R H]
        tail_Matrix = head_Matrix + relation_Matrix

        return head_e, tail_Matrix

    def _memory_update_cell(self, user_memory, update_memory):
        z = torch.sigmoid(
            torch.mul(user_memory, update_memory).sum(-1).float()
        ).unsqueeze(
            -1
        )  # [B R 1], the gate vector
        updated_user_memory = (1.0 - z) * user_memory + z * update_memory
        return updated_user_memory

    def memory_update(self, item_seq, item_seq_len,disen_weight):

        """define write operator"""
        step_length = item_seq.size()[1]
        last_item = item_seq_len - 1
        # init user memory with 0s
        user_memory = (
            # torch.zeros(item_seq.size()[0], self.n_relations, self.embedding_size)
            torch.zeros(item_seq.size()[0], self.n_factors, self.embedding_size)
                .float()
            .to(self.device)
        )  # [B R H]
        last_user_memory = torch.zeros_like(user_memory)
        for i in range(step_length):  # [len]
            _, update_memory = self._get_kg_embedding(item_seq[:, i],disen_weight)  # [B R H]
            user_memory = self._memory_update_cell(
                user_memory, update_memory
            )  # [B R H]
            last_user_memory[last_item == i] = user_memory[last_item == i].float()
        return last_user_memory

    def memory_read(self, user_memory,disen_weight):
        """define read operator"""
        attrs = disen_weight
        attentions = nn.functional.softmax(
            self.gamma * torch.mul(user_memory, attrs).sum(-1).float(), -1
        )  # [B R]
        u_m = torch.mul(user_memory, attentions.unsqueeze(-1)).sum(1)  # [B H]
        return u_m

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)[-1]

        # output = trm_output[-1]


        disentangled_intention_emb1 = self.disentangle_encoder(trm_output, item_seq_len)  # [B, K, L, H]

        gather_index = item_seq_len.view(-1, 1, 1, 1).repeat(1, self.k_intention, 1, self.embedding_size)

        disentangled_intention_emb = disentangled_intention_emb1.gather(2, gather_index - 1).squeeze()  # [B, K, H]

        return disentangled_intention_emb, disentangled_intention_emb1


    def forward2(self,output,item_seq, item_seq_len):

        disen_weight = torch.mm(
            nn.Softmax(dim=-1)(self.disen_weight_att), self.relation_Matrix.to(torch.float32)
        )  # [n_factors, embedding_size]

        # attribute-based preference representation, m^u_t
        user_memory = self.memory_update(item_seq, item_seq_len, disen_weight)

        u_m = self.memory_read(user_memory, disen_weight)
        u_m = u_m.unsqueeze(1).repeat(1, output.size(1), 1)  # 复制为[B, n_items, H]
        # combine them together
        p_u = self.dense_layer_u(torch.cat((output, u_m), -1))  # [B, n_items,  H]


        return p_u, disen_weight


    def _get_item_comb_embedding(self, item,disen_weight):
        h_e, _ = self._get_kg_embedding(item,disen_weight)
        i_e = self.item_embedding(item)
        q_i = self.dense_layer_i(torch.cat((i_e, h_e), -1))  # [B H]
        return q_i

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        # pretrain
        if self.train_stage == "pretrain":
            disentangled_intention_emb,aug_disentangled_intention_emb1 = self.forward(item_seq, item_seq_len)
            pos_items = interaction[self.POS_ITEM_ID]
            if self.loss_type == 'BPR':
                neg_items = interaction[self.NEG_ITEM_ID]
                pos_items_emb = self.item_embedding(pos_items)
                neg_items_emb = self.item_embedding(neg_items)
                pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
                neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
                loss = self.loss_fct(pos_score, neg_score)
            else:  # self.loss_type = 'CE'
                test_item_emb = self.item_embedding.weight
                if test_item_emb.ndim == 2:
                    test_item_emb = test_item_emb.unsqueeze(0)  # [1, num_items, H]
                logits = disentangled_intention_emb @ test_item_emb.permute(0, 2, 1)  # [B, K, num_items]
                if self.score_type == 'max':
                    logits, max_indices = torch.max(logits, 1)  # [B, num_items]
                elif self.score_type == 'mean':
                    logits = logits.mean(dim=1)  # [B, num_items]
                else:
                    raise ValueError('Invalid method of aggregating matching score')
                loss = self.loss_fct(logits, pos_items)

            # Unsupervised NCE
            if self.ssl in ['x']:
                B = pos_items.size(0)
                aug_disentangled_intention_1 = aug_disentangled_intention_emb1  # [B, K, L, D]
                aug_disentangled_intention_1 = aug_disentangled_intention_1.view(B * self.k_intention,
                                                                                 -1)  # [B * K, L * D]
                _, aug_disentangled_intention_2 = self.forward(item_seq, item_seq_len)  # [B, K, L, D]
                aug_disentangled_intention_2 = aug_disentangled_intention_2.view(B * self.k_intention,
                                                                                 -1)  # [B * K, L * D]
                loss += self.lmd * self.nce_loss(aug_disentangled_intention_1, aug_disentangled_intention_2)

            # Supervised NCE
            if self.ssl in ['us', 'su']:
                sem_aug, sem_aug_lengths = interaction['sem_aug'], interaction['sem_aug_lengths']
                sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths)

                sem_nce_logits, sem_nce_labels = self.info_nce(seq_output, sem_aug_seq_output, temp=self.tau,
                                                               batch_size=item_seq_len.shape[0], sim=self.sim)

                loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)

            if self.ssl == 'us_x':
                aug_seq_output = self.forward(item_seq, item_seq_len)

                sem_aug, sem_aug_lengths = interaction['sem_aug'], interaction['sem_aug_lengths']
                sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths)

                sem_nce_logits, sem_nce_labels = self.info_nce(aug_seq_output, sem_aug_seq_output, temp=self.tau,
                                                               batch_size=item_seq_len.shape[0], sim=self.sim)

                loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)

        # finetune
        else:
            pos_items = interaction[self.POS_ITEM_ID]
            disentangled_intention_emb, _ = self.forward(item_seq, item_seq_len)
            if self.loss_type == "BPR":
                neg_items = interaction[self.NEG_ITEM_ID]
                pos_items_emb = self._get_item_comb_embedding(pos_items, disen_weight)
                neg_items_emb = self._get_item_comb_embedding(neg_items, disen_weight)
                pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
                neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
                loss = self.loss_fct(pos_score, neg_score)
            else:  # self.loss_type = 'CE'
                # 将这快进行修改
                test_items_emb = self.item_embedding.weight   # [num_items, H]
                test_entity_emb = self.entity_embedding.weight   # [num_items, H]
                num_items = test_items_emb.size(0)
                batch_size = 1000
                logit = []
                for i in range(0, num_items, batch_size):

                    batch_items_emb = test_items_emb[i:i + batch_size, :]  # [batch_size_num_items, H]

                    batch_logits = disentangled_intention_emb @ batch_items_emb.permute(1, 0)  # [B, K, batch_size_num_items]

                    if self.score_type == 'max':
                        batch_logits, max_indices = torch.max(batch_logits, 1)  # [B, batch_size_num_items]
                    elif self.score_type == 'mean':
                        batch_logits = batch_logits.mean(dim=1)  # [B, batch_size_num_items]
                    else:
                        raise ValueError('Invalid method of aggregating matching score')

                    max_disentangled_intention_emb = disentangled_intention_emb.gather(1,max_indices.unsqueeze(-1).expand(-1, -1,self.embedding_size))  # [B,batch_size_num_items,D]

                    seq_output, disen_weight = self.forward2(max_disentangled_intention_emb, item_seq,
                                                             item_seq_len)  # seq_output应该是[B, batch_size_num_items, H]

                    batch_entity_emb = test_entity_emb[i:i + batch_size, :]
                    batch_test_items_emb = self.dense_layer_i(
                        torch.cat((batch_items_emb, batch_entity_emb), -1)
                    )  # [batch_size_num_items H]

                    batch_logit = torch.matmul(seq_output, batch_test_items_emb.transpose(0, 1)).diagonal(dim1=-2,dim2=-1)  # [B, batch_size_num_items, batch_size_num_items]  ->[B, batch_size_num_items]

                    logit.append(batch_logit)

                logits = torch.cat(logit, dim=1)

                loss = self.loss_fct(logits, pos_items)
            loss = loss + self.sim_decay * self.calculate_cor_loss(self.disen_weight_att)
        return loss

    def calculate_cor_loss(self, tensors):
        def CosineSimilarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = F.normalize(tensor_1, dim=0)
            normalized_tensor_2 = F.normalize(tensor_2, dim=0)
            return (normalized_tensor_1 * normalized_tensor_2).sum(
                dim=0
            ) ** 2  # no negative

        def DistanceCorrelation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = torch.zeros(channel, channel).to(tensor_1.device)
            zero = torch.zeros(1).to(tensor_1.device)
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)
            """cul distance matrix"""
            a_, b_ = (
                torch.matmul(tensor_1, tensor_1.t()) * 2,
                torch.matmul(tensor_2, tensor_2.t()) * 2,
            )  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1**2, tensor_2**2
            a, b = torch.sqrt(
                torch.max(tensor_1_square - a_ + tensor_1_square.t(), zeros) + 1e-8
            ), torch.sqrt(
                torch.max(tensor_2_square - b_ + tensor_2_square.t(), zeros) + 1e-8
            )  # [channel, channel]
            """cul distance correlation"""
            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.max((A * B).sum() / channel**2, zero) + 1e-8)
            dcov_AA = torch.sqrt(torch.max((A * A).sum() / channel**2, zero) + 1e-8)
            dcov_BB = torch.sqrt(torch.max((B * B).sum() / channel**2, zero) + 1e-8)
            return dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)

        def MutualInformation(tensors):
            # tensors: [n_factors, dimension]
            # normalized_tensors: [n_factors, dimension]
            normalized_tensors = F.normalize(tensors, dim=1)
            scores = torch.mm(normalized_tensors, normalized_tensors.t())
            scores = torch.exp(scores / self.temperature)
            cor_loss = -torch.sum(torch.log(scores.diag() / scores.sum(1)))
            return cor_loss

        """cul similarity for each latent factor weight pairs"""
        if self.ind == "mi":
            return MutualInformation(tensors)
        elif self.ind == "distance":
            cor_loss = 0.0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    cor_loss += DistanceCorrelation(tensors[i], tensors[j])
        elif self.ind == "cosine":
            cor_loss = 0.0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    cor_loss += CosineSimilarity(tensors[i], tensors[j])
        else:
            raise NotImplementedError(
                f"The independence loss type [{self.ind}] has not been supported."
            )
        return cor_loss


    def predict(self, interaction):
        if self.train_stage == "pretrain":
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            test_item = interaction[self.ITEM_ID]
            disentangled_intention_emb, _ = self.forward(item_seq, item_seq_len)
            test_item_emb = self.item_embedding(test_item)
            if test_item_emb.ndim == 2:
                test_item_emb = test_item_emb.unsqueeze(0)  # [1, num_items, H]
            logits = disentangled_intention_emb @ test_item_emb.permute(0, 2, 1)
            # print('logits.shape:', logits.shape)   # torch.Size([1, 4, 1])   [B ,K ,num_items]
            if self.score_type == 'max':
                logits, max_indices = torch.max(logits, 1)  # [B, num_items]
            elif self.score_type == 'mean':
                logits = logits.mean(dim=1)  # [B, num_items]
            else:
                raise ValueError('Invalid method of aggregating matching score')
            scores = logits.sum(dim=1)

        else:
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            test_item = interaction[self.ITEM_ID]
            # _,seq_output = self.forward(item_seq, item_seq_len, test_item, candidate_aware=True)
            disentangled_intention_emb, _ = self.forward(item_seq, item_seq_len)  # [K, H]

            test_item_emb = self.item_embedding(test_item)   # [1, H]
            if test_item_emb.ndim == 2:
                test_item_emb = test_item_emb.unsqueeze(0)  # [1, num_items, H]
            logits = disentangled_intention_emb @ test_item_emb.permute(0, 2, 1)  # [1, K, num_items]
            if self.score_type == 'max':
                logits, max_indices = torch.max(logits, 1)  # [1, 1]
            elif self.score_type == 'mean':
                logits = logits.mean(dim=1)  # [1, 1]
            else:
                raise ValueError('Invalid method of aggregating matching score')
            disentangled_intention_emb = disentangled_intention_emb.unsqueeze(0) # [1, K, H]
            max_disentangled_intention_emb = disentangled_intention_emb.gather(1,max_indices.unsqueeze(-1).expand(-1, -1,
                                                                                                                self.embedding_size))  # [1,1,H]

            seq_output, disen_weight = self.forward2(max_disentangled_intention_emb, item_seq, item_seq_len)   # [1,num_items,H]

            test_item_emb = self._get_item_comb_embedding(test_item, disen_weight)

            scores = torch.mul(seq_output, test_item_emb).diagonal(dim1=-2, dim2=-1).sum(dim=1)  # [B,num_items,num_items] ->[B,num_items] -> [B]

        return scores

    def full_sort_predict(self, interaction):
        if self.train_stage == "pretrain":
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            disentangled_intention_emb, _ = self.forward(item_seq, item_seq_len)
            test_items_emb = self.item_embedding.weight
            
            if test_items_emb.ndim == 2:
                test_items_emb = test_items_emb.unsqueeze(0)
            scores = disentangled_intention_emb @ test_items_emb.permute(0, 2, 1)  # [B, K, num_items]
            if self.score_type == 'max':
                scores, max_indices = torch.max(scores, 1)  # [B, num_items]
            elif self.score_type == 'mean':
                scores = scores.mean(dim=1)  # [B, num_items]
            else:
                raise ValueError('Invalid method of aggregating matching score')

        else:
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            disentangled_intention_emb, _ = self.forward(item_seq, item_seq_len)  # [B, K, H]
            test_items_emb = self.item_embedding.weight  # [ num_items, H]
            test_entitys_emb = self.entity_embedding.weight
            num_items = test_items_emb.size(0)
            batch_size = 1000
            score = []
            for i in range(0, num_items, batch_size):
                batch_items_emb = test_items_emb[i:i + batch_size, :]    # [batch_size, H]
                batch_logits = disentangled_intention_emb @ batch_items_emb.permute(1,0)  # [B, K, batch_size]
                if self.score_type == 'max':
                    batch_logits, max_indices = torch.max(batch_logits, 1)  # [B, num_items]
                elif self.score_type == 'mean':
                    batch_logits = batch_logits.mean(dim=1)  # [B, num_items]
                else:
                    raise ValueError('Invalid method of aggregating matching score')
                max_disentangled_intention_emb = disentangled_intention_emb.gather(1,max_indices.unsqueeze(-1).expand(-1, -1,self.embedding_size))  # [B,num_items,D]
                seq_output, disen_weight = self.forward2(max_disentangled_intention_emb, item_seq,
                                                         item_seq_len)  # seq_output应该是[B, n_items, H]
                batch_entity_emb = test_entitys_emb[i:i + batch_size, :]
                batch_test_items_emb = self.dense_layer_i(
                    torch.cat((batch_items_emb, batch_entity_emb), -1)
                )  # [n_items H]
                batch_score = torch.matmul(seq_output, batch_test_items_emb.transpose(0, 1)).diagonal(dim1=-2, dim2=-1)  # [B, n_items, n_items]  ->[B, n_items]

                score.append(batch_score)
            scores = torch.cat(score, dim=1)  # [B,H]
        return scores


    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        # pairwise l2 distace
        sim = torch.cdist(z, z, p=2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distace
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity



class InfoNCELoss(nn.Module):
    """
    Pair-wise Noise Contrastive Estimation Loss
    """

    def __init__(self, temperature, similarity_type):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature  # temperature
        self.sim_type = similarity_type  # cos or dot
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, aug_hidden_view1, aug_hidden_view2, mask=None):
        """
        Args:
            aug_hidden_view1 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1
            aug_hidden_view2 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1

        Returns: nce_loss (FloatTensor, (,)): calculated nce loss
        """
        if aug_hidden_view1.ndim > 2:
            # flatten tensor
            aug_hidden_view1 = aug_hidden_view1.view(aug_hidden_view1.size(0), -1)
            aug_hidden_view2 = aug_hidden_view2.view(aug_hidden_view2.size(0), -1)

        if self.sim_type not in ['cos', 'dot']:
            raise Exception(f"Invalid similarity_type for cs loss: [current:{self.sim_type}]. "
                            f"Please choose from ['cos', 'dot']")

        if self.sim_type == 'cos':
            sim11 = self.cosinesim(aug_hidden_view1, aug_hidden_view1)
            sim22 = self.cosinesim(aug_hidden_view2, aug_hidden_view2)
            sim12 = self.cosinesim(aug_hidden_view1, aug_hidden_view2)
        elif self.sim_type == 'dot':
            # calc similarity
            sim11 = aug_hidden_view1 @ aug_hidden_view1.t()
            sim22 = aug_hidden_view2 @ aug_hidden_view2.t()
            sim12 = aug_hidden_view1 @ aug_hidden_view2.t()
        # mask non-calc value
        sim11[..., range(sim11.size(0)), range(sim11.size(0))] = float('-inf')
        sim22[..., range(sim22.size(0)), range(sim22.size(0))] = float('-inf')

        cl_logits1 = torch.cat([sim12, sim11], -1)
        cl_logits2 = torch.cat([sim22, sim12.t()], -1)
        cl_logits = torch.cat([cl_logits1, cl_logits2], 0) / self.temperature
        if mask is not None:
            cl_logits = torch.masked_fill(cl_logits, mask, float('-inf'))
        target = torch.arange(cl_logits.size(0)).long().to(aug_hidden_view1.device)
        cl_loss = self.criterion(cl_logits, target)

        return cl_loss

    def cosinesim(self, aug_hidden1, aug_hidden2):
        h = torch.matmul(aug_hidden1, aug_hidden2.T)
        h1_norm2 = aug_hidden1.pow(2).sum(dim=-1).sqrt().view(h.shape[0], 1)
        h2_norm2 = aug_hidden2.pow(2).sum(dim=-1).sqrt().view(1, h.shape[0])
        return h / (h1_norm2 @ h2_norm2)


class DisentangleEncoder(nn.Module):
    def __init__(self, k_intention, embed_size, max_len):
        super(DisentangleEncoder, self).__init__()
        self.embed_size = embed_size

        self.intentions = nn.Parameter(torch.randn(k_intention, embed_size))
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.layer_norm_3 = nn.LayerNorm(embed_size)


    def forward(self, local_item_emb, seq_len):
        """
        Args:
            local_item_emb: [B, L, D]
            global_item_emb: [B, L, D]
            seq_len: [B]
        Returns:
            disentangled_intention_emb: [B, K, L, D]
        """
        disentangled_intention_emb = self.intention_disentangling(local_item_emb, seq_len)

        return disentangled_intention_emb

    def item2IntentionScore(self, item_emb):
        """
        Args:
            item_emb: [B, L, D]
        Returns:
            score: [B, L, K]  L个项目在K个意图上的得分
        """
        item_emb_norm = self.layer_norm_1(item_emb)  # [B, L, D]
        intention_norm = self.layer_norm_2(self.intentions).unsqueeze(0)  # [1, K, D]

        logits = item_emb_norm @ intention_norm.permute(0, 2, 1)  # [B, L, K]
        score = F.softmax(logits / math.sqrt(self.embed_size), -1)

        return score


    def intention_disentangling(self, item_emb, seq_len):
        """
        Args:
            item_emb: [B. L, D]
            seq_len: [B]
        Returns:
            item_disentangled_emb: [B, K, L, D]
        """
        # get score
        item2intention_score = self.item2IntentionScore(item_emb)
        # item_attn_weight = self.item2AttnWeight(item_emb, seq_len)

        # get disentangled embedding
        # score_fuse = item2intention_score * item_attn_weight.unsqueeze(-1)  # [B, L, K]
        score_fuse = item2intention_score.permute(0, 2, 1).unsqueeze(-1)  # [B, K, L, 1]
        item_emb_k = item_emb.unsqueeze(1)  # [B, 1, L, D]
        disentangled_item_emb = self.layer_norm_3(score_fuse * item_emb_k)
        return disentangled_item_emb




