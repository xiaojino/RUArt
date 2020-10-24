import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn.parameter import Parameter
from Models.Bert.Bert import Bert
from Models.Layers import MaxPooling, CNN, dropout, RNN_from_opt, set_dropout_prob, weighted_avg, set_seq_dropout, Attention, DeepAttention, LinearSelfAttn, GetFinalScores
from Utils.CoQAUtils import POS, ENT
from copy import deepcopy
import logging
log = logging.getLogger(__name__)
from torch.nn.utils.weight_norm import weight_norm
'''
 SDNet
'''
class SDNet(nn.Module):
    def __init__(self, opt, embedding):
        super(SDNet, self).__init__()
        print('SDNet model\n')
        self.opt = opt
        self.vocab_dim = 300
        if 'PHOC' in self.opt:
            phoc_embedding = embedding['phoc_embedding']
        if 'FastText' in self.opt:
            fast_embedding = embedding['fast_embedding']
        if 'GLOVE' in self.opt:
            glove_embedding = embedding['glove_embedding']
        if 'ModelParallel' in self.opt:
            self.bert_cuda = 'cuda:{}'.format(self.opt['ModelParallel'][-1])
            self.main_cuda = 'cuda:{}'.format(self.opt['ModelParallel'][0])
        #self.position_dim = opt['position_dim']
        self.use_cuda = (self.opt['cuda'] == True)
        self.q_embedding = opt['q_embedding'].split(',')
        self.ocr_embedding = opt['ocr_embedding'].split(',')
        self.LN_flag = 'LN' in self.opt
        if self.LN_flag:
            log.info('Do Layer Normalization')
        else:
            log.info('Do not do Layer Normalization')

        set_dropout_prob(0.0 if not 'DROPOUT' in opt else float(opt['DROPOUT']))
        set_seq_dropout('VARIATIONAL_DROPOUT' in self.opt)

        x_input_size = 0
        ques_input_size = 0

        if 'PHOC' in self.opt:
            self.vocab_size = int(opt['vocab_size'])
            self.phoc_dim = int(opt['phoc_dim'])
            self.phoc_embed = nn.Embedding(self.vocab_size, self.phoc_dim, padding_idx = 1)
            self.phoc_embed.weight.data = phoc_embedding

        if 'FastText' in self.opt:
            self.vocab_size = int(opt['vocab_size'])
            self.fast_dim = int(opt['fast_dim'])
            self.fast_embed = nn.Embedding(self.vocab_size, self.fast_dim, padding_idx = 1)
            self.fast_embed.weight.data = fast_embedding

        if 'GLOVE' in self.opt:
            self.vocab_size = int(opt['vocab_size'])
            self.glove_dim = int(opt['glove_dim'])
            self.glove_embed = nn.Embedding(self.vocab_size, self.glove_dim, padding_idx = 1)
            self.glove_embed.weight.data = glove_embedding

        x_input_size += self.glove_dim if 'glove' in self.ocr_embedding else 0
        ques_input_size += self.glove_dim if 'glove' in self.q_embedding else 0
        x_input_size += self.fast_dim if 'fasttext' in self.ocr_embedding else 0
        ques_input_size += self.fast_dim if 'fasttext' in self.q_embedding else 0
        x_input_size += self.phoc_dim if 'phoc' in self.ocr_embedding else 0
        ques_input_size += self.phoc_dim if 'phoc' in self.q_embedding else 0

        if 'TUNE_PARTIAL' in self.opt:
            print('TUNE_PARTIAL')
            if 'FastText' in self.opt:
                self.fixed_embedding_fast = fast_embedding[opt['tune_partial']:]
            if 'GLOVE' in self.opt:
                self.fixed_embedding_glove = glove_embedding[opt['tune_partial']:]
        else:
            if 'FastText' in self.opt:
                self.fast_embed.weight.requires_grad = False
            if 'GLOVE' in self.opt:
                self.glove_embed.weight.requires_grad = False

        if 'BERT' in self.opt:
            print('Using BERT')
            self.Bert = Bert(self.opt)
            if 'LOCK_BERT' in self.opt:
                print('Lock BERT\'s weights')
                for p in self.Bert.parameters():
                    p.requires_grad = False
            if 'BERT_LARGE' in self.opt:
                print('BERT_LARGE')
                bert_dim = 1024
                bert_layers = 24
            else:
                bert_dim = 768
                bert_layers = 12

            print('BERT dim:', bert_dim, 'BERT_LAYERS:', bert_layers)    

            if 'BERT_LINEAR_COMBINE' in self.opt:
                print('BERT_LINEAR_COMBINE')
                self.alphaBERT = nn.Parameter(torch.Tensor(bert_layers), requires_grad=True)
                self.gammaBERT = nn.Parameter(torch.Tensor(1, 1), requires_grad=True)
                torch.nn.init.constant_(self.alphaBERT, 1.0)
                torch.nn.init.constant_(self.gammaBERT, 1.0)
                
            cdim = bert_dim
            x_input_size += bert_dim if 'bert' in self.ocr_embedding or 'bert_only' in self.ocr_embedding else 0
            ques_input_size += bert_dim if 'bert' in self.q_embedding or 'bert_only' in self.q_embedding else 0

        if 'PRE_ALIGN' in self.opt:
            self.pre_align = Attention(self.vocab_dim, opt['prealign_hidden'], correlation_func = 3, do_similarity = True)
            if 'PRE_ALIGN_befor_rnn' in self.opt:
                x_input_size += self.vocab_dim

        if 'pos' in self.q_embedding or 'pos' in self.ocr_embedding:
            pos_dim = opt['pos_dim']
            self.pos_embedding = nn.Embedding(len(POS), pos_dim)
            x_input_size += pos_dim if 'pos' in self.ocr_embedding else 0
            ques_input_size += pos_dim if 'pos' in self.q_embedding else 0
        if 'ent' in self.q_embedding or 'pos' in self.ocr_embedding:
            ent_dim = opt['ent_dim']
            self.ent_embedding = nn.Embedding(len(ENT), ent_dim)
            x_input_size += ent_dim if 'ent' in self.ocr_embedding else 0
            ques_input_size += ent_dim if 'ent' in self.q_embedding else 0
        

        print('Initially, the vector_sizes [ocr, query] are', x_input_size, ques_input_size)
        addtional_feat = 0
        self.LN = 'LN' in opt

        self.multi2one, multi2one_output_size = RNN_from_opt(x_input_size, opt['multi2one_hidden_size'],num_layers=1, concat_rnn=opt['concat_rnn'], add_feat=addtional_feat, bidirectional=self.opt['multi2one_bidir'])

        # if 'LN' in self.opt:
        #     self.ocr_input_ln = nn.LayerNorm([opt['batch_size'], opt['max_ocr_num'], multi2one_output_size])
        #     self.od_input_ln = nn.LayerNorm([opt['batch_size'], opt['max_od_num'], multi2one_output_size])
        
        self.multi2one_output_size = multi2one_output_size

        # RNN context encoder
        
        self.context_rnn, context_rnn_output_size = RNN_from_opt(multi2one_output_size, opt['hidden_size'], num_layers=opt['in_rnn_layers'], concat_rnn=opt['concat_rnn'], add_feat=addtional_feat)
        
        # RNN question encoder
        self.ques_rnn, ques_rnn_output_size = RNN_from_opt(ques_input_size, opt['hidden_size'], num_layers=opt['in_rnn_layers'], concat_rnn=opt['concat_rnn'], add_feat=addtional_feat)
        # if 'LN' in self.opt:
        #     self.ocr_rnn1_ln = nn.LayerNorm([opt['batch_size'], opt['max_ocr_num'], context_rnn_output_size])
        #     self.od_rnn1_ln = nn.LayerNorm([opt['batch_size'], opt['max_od_num'], context_rnn_output_size])
        #     self.q_rnn1_ln = nn.LayerNorm([opt['batch_size'], opt['max_od_num'], ques_rnn_output_size])

        # Output sizes of rnn encoders
        print('After Input LSTM, the vector_sizes [doc, query] are [', context_rnn_output_size, ques_rnn_output_size, '] *', opt['in_rnn_layers'])

        # Deep inter-attention
        if ('GLOVE' not in self.opt) and ('FastText' not in self.opt):
            _word_hidden_size = 0
        else:
            _word_hidden_size = multi2one_output_size + addtional_feat

        self.deep_attn = DeepAttention(opt, abstr_list_cnt=opt['in_rnn_layers'], deep_att_hidden_size_per_abstr=opt['deep_att_hidden_size_per_abstr'], correlation_func=3, word_hidden_size=_word_hidden_size)

        self.deep_attn_input_size = self.deep_attn.rnn_input_size
        self.deep_attn_output_size = self.deep_attn.output_size
        print('Deep Attention: input: {}, hidden input: {}, output: {}'.format(self.deep_attn.att_size, self.deep_attn_input_size, self.deep_attn_output_size))

        # Question understanding and compression
        self.high_lvl_ques_rnn , high_lvl_ques_rnn_output_size = RNN_from_opt(ques_rnn_output_size * opt['in_rnn_layers'], opt['highlvl_hidden_size'], num_layers = opt['question_high_lvl_rnn_layers'], concat_rnn = True)


        self.after_deep_attn_size = self.deep_attn_output_size + self.deep_attn_input_size + addtional_feat + multi2one_output_size
        self.self_attn_input_size = self.after_deep_attn_size
                

        # Self attention on context
        if 'no_Context_Self_Attention' in self.opt:
            print('no self attention on context')
            self_attn_output_size = 0
        else:
            self.highlvl_self_att = Attention(self.self_attn_input_size, opt['deep_att_hidden_size_per_abstr'], correlation_func=3)
            self_attn_output_size = self.deep_attn_output_size
            print('Self deep-attention input is {}-dim'.format(self.self_attn_input_size))

        self.high_lvl_context_rnn, high_lvl_context_rnn_output_size = RNN_from_opt(self.deep_attn_output_size + self_attn_output_size, opt['highlvl_hidden_size'], num_layers = 1, concat_rnn = False)
        context_final_size = high_lvl_context_rnn_output_size
        # if 'LN' in self.opt:
        #     self.ocr_rnn1_ln = nn.LayerNorm([opt['batch_size'], opt['max_ocr_num'], high_lvl_context_rnn_output_size])
        #     self.od_rnn1_ln = nn.LayerNorm([opt['batch_size'], opt['max_od_num'], high_lvl_context_rnn_output_size])
        #     self.q_rnn1_ln = nn.LayerNorm([opt['batch_size'], opt['max_od_num'], high_lvl_ques_rnn_output_size])

        print('Do Question self attention')
        self.ques_self_attn = Attention(high_lvl_ques_rnn_output_size, opt['query_self_attn_hidden_size'], correlation_func=3)
        
        ques_final_size = high_lvl_ques_rnn_output_size
        print('Before answer span finding, hidden size are', context_final_size, ques_final_size)

        
        if 'position_dim' in self.opt:
            if self.opt['position_mod'] == 'qk+':
                self.od_ocr_attn = Attention(context_final_size, opt['hidden_size'], correlation_func = 3, do_similarity = True)
                self.position_attn = Attention(self.opt['position_dim'], opt['hidden_size'], correlation_func = 3, do_similarity = True)
                position_att_output_size = context_final_size
            elif self.opt['position_mod'] == 'cat':
                self.od_ocr_attn = Attention(context_final_size+self.opt['position_dim'], opt['hidden_size'], correlation_func = 3, do_similarity = True)
                position_att_output_size = context_final_size + self.opt['position_dim']
        # Question merging
        self.ques_merger = LinearSelfAttn(ques_final_size)
        if self.opt['pos_att_merge_mod'] == 'cat':
            ocr_final_size = context_final_size + position_att_output_size
            # self.get_answer = GetFinalScores(context_final_size + position_att_output_size, ques_final_size)
        elif self.opt['pos_att_merge_mod'] == 'atted':
            ocr_final_size = position_att_output_size
            # self.get_answer = GetFinalScores(position_att_output_size, ques_final_size)
        elif self.opt['pos_att_merge_mod'] == 'original':
            ocr_final_size = context_final_size
            # self.get_answer = GetFinalScores(context_final_size, ques_final_size)
        if 'img_feature' in self.opt:
            if self.opt['img_fea_way'] == 'replace_od':
                self.img_fea_num = self.opt['img_fea_num']
                self.img_fea_dim = self.opt['img_fea_dim']
                self.img_spa_dim = self.opt['img_spa_dim']
                self.img_fea2od = nn.Linear(self.opt['img_fea_dim'], multi2one_output_size)
                # self.pro_que_rnn, pro_que_rnn_output_size = RNN_from_opt(ques_input_size, multi2one_output_size//2)
                # assert pro_que_rnn_output_size == multi2one_output_size
                # ques_input_size = multi2one_output_size
            elif self.opt['img_fea_way'] == 'final_att':
                self.img_fea_num = self.opt['img_fea_num']
                self.img_fea_dim = self.opt['img_fea_dim']
                self.img_spa_dim = self.opt['img_spa_dim']
                self.image_feature_model = Image_feature_model(ques_final_size, self.img_fea_dim)
                self.ocr_final_model = Image_feature_model(ques_final_size, ocr_final_size)
                self.fixed_ocr_alpha = nn.Parameter(torch.Tensor(1, 1), requires_grad=True)
                torch.nn.init.constant_(self.fixed_ocr_alpha, 0.5)
                ques_final_size += ques_final_size * 2
            else:
                assert False

        self.get_answer = GetFinalScores(ocr_final_size, ques_final_size, yesno='label_yesno' in self.opt, no_answer='label_no_answer' in self.opt, useES='useES' in self.opt)
        if 'fixed_answers' in self.opt:
            self.fixed_ans_classifier = Fixed_answers_predictor(ques_final_size, self.opt['fixed_answers_len'])

        if 'ES_ocr' in self.opt and self.opt['ES_using_way'] == 'post_process':
            self.ES_linear = nn.Linear(multi2one_output_size, ocr_final_size)
            self.ES_ocr_att = Attention(ocr_final_size, opt['hidden_size'], correlation_func = 3, do_similarity = True)
            # elif self.opt['ES_using_way'] == 'as_ocr':

        log.debug('Network build successes')

    def forward(self, q_list, ocr_list, od_list, return_score=False):
        if return_score:
            att_score = {}
        else:
            att_score = None
        batch_size = len(ocr_list['num_cnt'])
        od_max_num = od_list['position'].size(1)
        ocr_max_num = ocr_list['position'].size(1)
        q_input = self.get_embedding_from_list(q_list, self.q_embedding, self.opt['q_emb_initial'])
        ocr_input = self.get_embedding_from_list(ocr_list, self.ocr_embedding, self.opt['ocr_emb_initial'])
        od_input = self.get_embedding_from_list(od_list, self.ocr_embedding, self.opt['ocr_emb_initial'])

        if 'PRE_ALIGN_befor_rnn' in self.opt:
            ocr_prealign, od_prealign = self.get_prealign_emb(q_list, ocr_list, od_list, batch_size)
            ocr_input = torch.cat([ocr_input, ocr_prealign], dim=-1)
            od_input = torch.cat([od_input, od_prealign], dim=-1)
        if 'fasttext' in self.opt['ocr_embedding']:
            multi2one_ocr_input = self.multi2one(ocr_input, ocr_list['fasttext_mask'])
            multi2one_od_input = self.multi2one(od_input, od_list['fasttext_mask'])
        elif 'glove' in self.opt['ocr_embedding']:
            multi2one_ocr_input = self.multi2one(ocr_input, ocr_list['glove_mask'])
            multi2one_od_input = self.multi2one(od_input, od_list['glove_mask'])

        if 'img_feature' in self.opt:
            img_fea = q_list['img_features']
            img_spa = q_list['img_spatials']
            if self.opt['img_fea_way'] == 'replace_od':
                od_input = self.img_fea2od(img_fea)
                od_mask = torch.ByteTensor(batch_size, self.img_fea_num).fill_(1).cuda()
            elif self.opt['img_fea_way'] == 'final_att':
                # img_fea = self.img_fea_linear(img_fea)
                od_input = torch.FloatTensor(batch_size, od_max_num, self.multi2one_output_size).fill_(0).cuda()
                od_mask = torch.ByteTensor(batch_size, od_max_num).fill_(0).cuda()
                # img_fea_mask = torch.ByteTensor(batch_size, self.img_fea_num).fill_(1).cuda()
        else:
            od_input = torch.FloatTensor(batch_size, od_max_num, self.multi2one_output_size).fill_(0).cuda()
            od_mask = torch.ByteTensor(batch_size, od_max_num).fill_(0).cuda()
        ocr_input = torch.FloatTensor(batch_size, ocr_max_num, self.multi2one_output_size).fill_(0).cuda()

        if 'ES_ocr' in self.opt and self.opt['ES_using_way'] == 'post_process':
            es_ocr_len = self.opt['ES_ocr_len']
            ocr_mask = torch.ByteTensor(batch_size, ocr_max_num-self.opt['ES_ocr_len']).fill_(0).cuda()
        else:
            es_ocr_len = None
            ocr_mask = torch.ByteTensor(batch_size, ocr_max_num).fill_(0).cuda()
        od_idx = ocr_idx = 0
        mask_copy = torch.ByteTensor(batch_size).fill_(0).cuda()
        for i in range(batch_size):
            if 'img_feature_replace_od' not in self.opt:
                od_cnt = 0
                for j in od_list['len_cnt'][i]:
                    od_input[i][od_cnt] = multi2one_od_input[od_idx][j-1]
                    od_cnt += 1
                    od_idx += 1
                od_mask[i][0:od_cnt] = 1
            ocr_cnt = 0
            for j in ocr_list['len_cnt'][i]:
                ocr_input[i][ocr_cnt] = multi2one_ocr_input[ocr_idx][j-1]
                ocr_cnt += 1
                ocr_idx += 1
            if es_ocr_len != None and ocr_cnt >= es_ocr_len and self.opt['ES_using_way'] == 'post_process':
                ocr_mask[i][0:ocr_cnt-es_ocr_len] = 1
                # o_mask_pre[i][0:ocr_cnt-101] = 0
            else:
                ocr_mask[i][0:ocr_cnt] = 1
                mask_copy[i] = ocr_cnt

        if es_ocr_len != None and self.opt['ES_using_way'] == 'post_process':
            es_emb = ocr_input[:, :es_ocr_len]
            ocr_input = ocr_input[:, es_ocr_len:]
            ocr_list['position'] = ocr_list['position'][:, es_ocr_len:]
            es_mask = torch.ByteTensor(batch_size, es_ocr_len).fill_(1).cuda()

        if 'fasttext' in self.opt['q_embedding']:
            q_mask = q_list['fasttext_mask']
        else:
            q_mask = q_list['glove_mask']
        if 'PRE_ALIGN_after_rnn' in self.opt:
            if 'fasttext' in self.opt['q_embedding']:
                ocr_prealign, ocr_word_leve_attention_score = self.pre_align(ocr_input, q_list['fasttext_emb'], q_mask)
                od_prealign, od_word_leve_attention_score = self.pre_align(od_input, q_list['fasttext_emb'], q_mask)
            else:
                ocr_prealign, ocr_word_leve_attention_score = self.pre_align(ocr_input, q_list['glove_emb'], q_mask)
                od_prealign, od_word_leve_attention_score = self.pre_align(od_input, q_list['glove_emb'], q_mask)

        _, ocr_rnn_layers = self.context_rnn(ocr_input, ocr_mask, return_list=True, x_additional=None, LN=True) # layer x batch x x_len x context_rnn_output_size
        _, q_rnn_layers = self.ques_rnn(q_input, q_mask, return_list=True, x_additional=None, LN=True) # layer x batch x q_len x ques_rnn_output_size
        _, od_rnn_layers = self.context_rnn(od_input, od_mask, return_list=True, x_additional=None, LN=True)
        # if 'LN' in self.opt:
        #     for i in range(len(ocr_rnn_layers)):
        #         ocr_rnn_layers[i] = self.ocr_rnn1_ln(ocr_rnn_layers[i])
        #     for i in range(len(od_rnn_layers)):
        #         od_rnn_layers[i] = self.od_rnn1_ln(od_rnn_layers[i])
        #     for i in range(len(q_rnn_layers)):
        #         q_rnn_layers[i] = self.q_rnn1_ln(q_rnn_layers[i])
        
        # rnn with question only 
        q_highlvl = self.high_lvl_ques_rnn(torch.cat(q_rnn_layers, 2), q_mask, LN=True) # batch x q_len x high_lvl_ques_rnn_output_size
        # if 'LN' in self.opt:
        #     q_highlvl = self.q_
        q_rnn_layers.append(q_highlvl) # (layer + 1) layers
        
        # deep multilevel inter-attention
        
        if 'GLOVE' not in self.opt and 'FastText' not in self.opt:
            ocr_long = []
            q_long = []
            od_long = []
        elif 'PRE_ALIGN_after_rnn' in self.opt:
            ocr_long = [ocr_prealign]
            if 'fasttext' in self.opt['q_embedding']:
                q_long = [q_list['fasttext_emb']]
            else:
                q_long = [q_list['glove_emb']]
            od_long = [od_prealign]
        else:
            ocr_long = [ocr_input]
            if 'fasttext' in self.opt['q_embedding']:
                q_long = [q_list['fasttext_emb']]
            else:
                q_long = [q_list['glove_emb']]
            od_long = [od_input]

        ocr_rnn_after_inter_attn, ocr_inter_attn = self.deep_attn(ocr_long, ocr_rnn_layers, q_long, q_rnn_layers, ocr_mask, q_mask, return_bef_rnn=True)
        od_rnn_after_inter_attn, od_inter_attn = self.deep_attn(od_long, od_rnn_layers, q_long, q_rnn_layers, od_mask, q_mask, return_bef_rnn=True)

        # deep self attention
        ocr_self_attn_input = torch.cat([ocr_rnn_after_inter_attn, ocr_inter_attn, ocr_input], 2)
        od_self_attn_input = torch.cat([od_rnn_after_inter_attn, od_inter_attn, od_input], 2)
        
        if 'no_Context_Self_Attention' in self.opt:
            ocr_highlvl_output = self.high_lvl_context_rnn(ocr_rnn_after_inter_attn, ocr_mask, LN=True)
            od_highlvl_output = self.high_lvl_context_rnn(od_rnn_after_inter_attn, od_mask, LN=True)
        else:
            ocr_self_attn_output = self.highlvl_self_att(ocr_self_attn_input, ocr_self_attn_input, ocr_mask, x3=ocr_rnn_after_inter_attn, drop_diagonal=False)
            od_self_attn_output = self.highlvl_self_att(od_self_attn_input, od_self_attn_input, od_mask, x3=od_rnn_after_inter_attn, drop_diagonal=False)
            ocr_highlvl_output = self.high_lvl_context_rnn(torch.cat([ocr_rnn_after_inter_attn, ocr_self_attn_output], 2), ocr_mask, LN=True)
            od_highlvl_output = self.high_lvl_context_rnn(torch.cat([od_rnn_after_inter_attn, od_self_attn_output], 2), od_mask, LN=True)

            
        if 'position_dim' in self.opt:
            ocr_position = ocr_list['position']
            od_position = od_list['position']
            if 'img_feature' in self.opt and self.opt['img_fea_way'] == 'replace_od':
                od_position = img_spa
            if self.opt['position_mod'] == 'qk+':
                x_od_ocr = self.od_ocr_attn(ocr_highlvl_output, od_highlvl_output, od_mask)
                pos_att = self.position_attn(ocr_position, od_position, od_mask, x3 = od_highlvl_output)
                x_od_ocr += pos_att
            elif self.opt['position_mod'] == 'cat':
                x_od_ocr = self.od_ocr_attn(torch.cat([ocr_highlvl_output, ocr_position],dim=2), torch.cat([od_highlvl_output, od_position],dim=2), od_mask)
        if self.opt['pos_att_merge_mod'] == 'cat':
            ocr_final = torch.cat([ocr_highlvl_output, x_od_ocr], 2)
        elif self.opt['pos_att_merge_mod'] == 'atted':
            ocr_final = x_od_ocr
        elif self.opt['pos_att_merge_mod'] == 'original':
            ocr_final = ocr_highlvl_output
        # question self attention  
        q_final = self.ques_self_attn(q_highlvl, q_highlvl, q_mask, drop_diagonal=False) # batch x q_len x high_lvl_ques_rnn_output_size

        # merge questions  
        q_merge_weights = self.ques_merger(q_final, q_mask) 
        q_merged = weighted_avg(q_final, q_merge_weights) # batch x ques_final_size

        # predict scores
        if es_ocr_len != None and self.opt['ES_using_way'] == 'post_process':
            es_mid = self.ES_linear(es_emb)
            es_final = self.ES_ocr_att(es_mid, ocr_final, ocr_mask)
            ocr_final = torch.cat([es_final, ocr_final], dim=-2)
            ocr_mask = torch.cat([es_mask, ocr_mask], dim=-1)
        if 'img_feature' in self.opt and self.opt['img_fea_way'] == 'final_att':
            img_fea = self.image_feature_model(q_merged, img_fea)
            # q_merged = torch.cat([q_merged, img_fea], dim=-1)
            #ocr_fea = self.ocr_final_model(q_merged, ocr_final, mask=ocr_mask)
            #q_merged = torch.cat([q_merged, ocr_fea, img_fea], dim=-1)
        if 'useES' in self.opt:
            score_s = self.get_answer(ocr_final, q_merged, ocr_mask, self.opt['ES_ocr_len'], mask_flag='mask_score' in self.opt)
        else: 
            score_s = self.get_answer(ocr_final, q_merged, ocr_mask, None, mask_flag='mask_score' in self.opt)
        if 'fixed_answers' in self.opt:
            fixed_ans_logits = self.fixed_ans_classifier(q_merged)
            fixed_ans_logits = self.fixed_ocr_alpha * fixed_ans_logits
            score_s = (1 - self.fixed_ocr_alpha) * score_s
            score_s = torch.cat([fixed_ans_logits, score_s], dim=-1)
        return score_s, att_score

    def get_embedding_from_list(self, item_list, embedding_names, initial_embed):
        emb_list = []
        if 'phoc' in embedding_names: 
            phoc_emb = self.phoc_embed(item_list['phoc'])
            if 'dropout_emb' in self.opt:
                emb_list.append(dropout(phoc_emb, p=self.opt['dropout_emb'], training=self.drop_emb))
            else:
                emb_list.append(phoc_emb)
        if 'fasttext' in embedding_names:
            fast_emb = self.fast_embed(item_list['fasttext'])
            if 'PRE_ALIGN_befor_rnn' in self.opt:
                item_list['fasttext_emb'] = fast_emb
            if 'dropout_emb' in self.opt:
                emb_list.append(dropout(fast_emb, p=self.opt['dropout_emb'], training=self.drop_emb))
            else:
                emb_list.append(fast_emb)

        if 'glove' in embedding_names:
            glove_emb = self.glove_embed(item_list['glove'])
            if 'PRE_ALIGN_befor_rnn' in self.opt:
                item_list['glove_emb'] = glove_emb
            if 'dropout_emb' in self.opt:
                emb_list.append(dropout(glove_emb, p=self.opt['dropout_emb'], training=self.drop_emb))
            else:
                emb_list.append(glove_emb)
        for k in ['bert', 'bert_only']:
            if k in embedding_names:
                if 'ModelParallel' in self.opt:
                    bert_cuda = self.bert_cuda
                    main_cuda = self.main_cuda
                    if k == 'bert':
                        if 'fasttext' == initial_embed:
                            bert_output = self.Bert(item_list['bert'], item_list['bert_mask'], item_list['bert_offsets'], item_list['fasttext_mask'].to(bert_cuda), device=main_cuda)
                        else:
                            bert_output = self.Bert(item_list['bert'], item_list['bert_mask'], item_list['bert_offsets'], item_list['glove_mask'].to(bert_cuda), device=main_cuda)
                else:
                    if k == 'bert':
                        if 'fasttext' == initial_embed:
                            bert_output = self.Bert(item_list['bert'], item_list['bert_mask'],
                                                    item_list['bert_offsets'], item_list['fasttext_mask'])
                        else:
                            bert_output = self.Bert(item_list['bert'], item_list['bert_mask'], item_list['bert_offsets'], item_list['glove_mask'])
                if 'BERT_LINEAR_COMBINE' in self.opt:
                    bert_output = self.linear_sum(bert_output, self.alphaBERT, self.gammaBERT)
                emb_list.append(bert_output)
        if 'pos' in embedding_names:
            emb_list.append(
                self.pos_embedding(item_list['pos'])
            )
        if 'ent' in embedding_names:
            emb_list.append(
                self.ent_embedding(item_list['ent'])
            )
        res = torch.cat(emb_list, dim=-1) # final embedding cat
        return res

    def get_prealign_emb(self, q_list, ocr_list, od_list, batch_size):
        ocr_token_num_max = od_token_num_max = -1
        ocr_st = od_st = 0
        for i in range(batch_size):
            od_token_num_max = max(od_token_num_max, sum(od_list['len_cnt'][i]))
            ocr_token_num_max = max(ocr_token_num_max, sum(ocr_list['len_cnt'][i]))
        od_prealign_word_embed = torch.FloatTensor(batch_size,od_token_num_max,300).fill_(0).cuda()
        ocr_prealign_word_embed = torch.FloatTensor(batch_size, ocr_token_num_max, 300).fill_(0).cuda()
        od_idx = ocr_idx = 0
        for i in range(batch_size):
            od_cnt = 0
            for j in od_list['len_cnt'][i]:
                if 'fasttext' in self.opt['ocr_embedding']:
                    od_prealign_word_embed[i][od_cnt:od_cnt+j] = od_list['fasttext_emb'][od_idx][:j]
                else:
                    od_prealign_word_embed[i][od_cnt:od_cnt + j] = od_list['glove_emb'][od_idx][:j]
                od_cnt += j
                od_idx += 1
            ocr_cnt = 0
            for j in ocr_list['len_cnt'][i]:
                if 'fasttext' in self.opt['ocr_embedding']:
                    ocr_prealign_word_embed[i][ocr_cnt:ocr_cnt+j] = ocr_list['fasttext_emb'][ocr_idx][:j]
                else:
                    ocr_prealign_word_embed[i][ocr_cnt:ocr_cnt + j] = ocr_list['glove_emb'][ocr_idx][:j]
                ocr_cnt += j
                ocr_idx += 1
        if 'fasttext' in self.opt['q_embedding']:
            ocr_prealign_glove = self.pre_align(ocr_prealign_word_embed, q_list['fasttext_emb'], q_list['fasttext_mask'])
            od_prealign_glove = self.pre_align(od_prealign_word_embed, q_list['fasttext_emb'], q_list['fasttext_mask'])
        else:
            ocr_prealign_glove = self.pre_align(ocr_prealign_word_embed, q_list['glove_emb'], q_list['glove_mask'])
            od_prealign_glove = self.pre_align(od_prealign_word_embed, q_list['glove_emb'], q_list['glove_mask'])

        if 'fasttext' in self.opt['ocr_embedding']:
            ocr_prealign = torch.FloatTensor(ocr_list['fasttext_emb'].size(0), ocr_list['fasttext_emb'].size(1),
                                             ocr_list['fasttext_emb'].size(2)).fill_(0).cuda()
            od_prealign = torch.FloatTensor(od_list['fasttext_emb'].size(0), od_list['fasttext_emb'].size(1),
                                            od_list['fasttext_emb'].size(2)).fill_(0).cuda()
        else:
            ocr_prealign = torch.FloatTensor(ocr_list['glove_emb'].size(0), ocr_list['glove_emb'].size(1),
                                             ocr_list['glove_emb'].size(2)).fill_(0).cuda()
            od_prealign = torch.FloatTensor(od_list['glove_emb'].size(0), od_list['glove_emb'].size(1),
                                            od_list['glove_emb'].size(2)).fill_(0).cuda()

        od_idx = ocr_idx = 0
        for i in range(batch_size):
            od_cnt = 0
            for j in od_list['len_cnt'][i]:
                od_prealign[od_idx][:j] = od_prealign_glove[i][od_cnt:od_cnt+j]
                od_cnt += j
                od_idx += 1
            ocr_cnt = 0
            for j in ocr_list['len_cnt'][i]:
                ocr_prealign[ocr_idx][:j] = ocr_prealign_glove[i][ocr_cnt:ocr_cnt+j]
                ocr_cnt += j
                ocr_idx += 1
        return ocr_prealign, od_prealign
        


    
    '''
     input: 
      x_char: batch x word_num x char_num
      x_char_mask: batch x word_num x char_num
     output: 
       x_char_cnn_final:  batch x word_num x char_cnn_hidden_size
    '''
    def character_cnn(self, x_char, x_char_mask):
        x_char_embed = self.char_embed(x_char) # batch x word_num x char_num x char_dim
        batch_size = x_char_embed.shape[0]
        word_num = x_char_embed.shape[1]
        char_num = x_char_embed.shape[2]
        char_dim = x_char_embed.shape[3]
        x_char_cnn = self.char_cnn(x_char_embed.contiguous().view(-1, char_num, char_dim), x_char_mask) # (batch x word_num) x char_num x char_cnn_hidden_size
        x_char_cnn_final = self.maxpooling(x_char_cnn, x_char_mask.contiguous().view(-1, char_num)).contiguous().view(batch_size, word_num, -1) # batch x word_num x char_cnn_hidden_size
        return x_char_cnn_final

    def linear_sum(self, output, alpha, gamma):
        alpha_softmax = F.softmax(alpha, dim=0)
        for i in range(len(output)):
            t = output[i] * alpha_softmax[i] * gamma
            if i == 0:
                res = t
            else:
                res += t

        res = dropout(res, p=self.opt['dropout_emb'], training=self.drop_emb)
        return res


class Image_feature_model(nn.Module):
    def __init__(self, ques_final_size, img_fea_size):
        super(Image_feature_model, self).__init__()
        self.linear = nn.Linear(img_fea_size, ques_final_size)
    def forward(self, que, img_fea, mask=None):
        img_fea = self.linear(img_fea)
        img_fea_weight = img_fea.bmm(que.unsqueeze(-1)).squeeze(-1)
        if mask != None:
            img_fea_weight.data.masked_fill_(~(mask.bool()), -float('inf'))
        img_fea_weight = F.softmax(img_fea_weight, dim=-1)
        averaged_img_fea = img_fea_weight.unsqueeze(-1).transpose(-1, -2).bmm(img_fea).squeeze(1)
        return averaged_img_fea
class Fixed_answers_predictor(nn.Module):
    def __init__(self, ques_final_size, fixed_answers_len):
        super(Fixed_answers_predictor, self).__init__()
        self.linear = nn.Linear(ques_final_size, fixed_answers_len+1)

    def forward(self, que_final):
        fixed_answers_logits = self.linear(que_final)
        fixed_answers_logits = F.softmax(fixed_answers_logits, dim=-1)
        return fixed_answers_logits

class LinearTransform(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearTransform, self).__init__()
        self.lc = weight_norm(
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=False), dim=None
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.lc(x)
