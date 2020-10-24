# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.weight_norm import weight_norm

def set_dropout_prob(p):
    global dropout_p
    dropout_p = p

def set_seq_dropout(option): # option = True or False
    global do_seq_dropout
    do_seq_dropout = option

def seq_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training == False or p == 0:
        return x
    dropout_mask = Variable(1.0 / (1-p) * torch.bernoulli((1-p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1)), requires_grad=False)
    return dropout_mask.unsqueeze(1).expand_as(x) * x    

def dropout(x, p=0, training=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if do_seq_dropout and len(x.size()) == 3: # if x is (batch * len * input_size)
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)

class CNN(nn.Module):
    def __init__(self, input_size, window_size, output_size):
        super(CNN, self).__init__()
        if window_size % 2 != 1:
            raise Exception("window size must be an odd number")
        padding_size = int((window_size - 1) / 2)
        self._output_size = output_size
        self.cnn = nn.Conv2d(1, output_size, (window_size, input_size), padding = (padding_size, 0), bias = False)
        init.xavier_uniform(self.cnn.weight)

    @property
    def output_size(self):
        return self._output_size

    '''
     (item, subitem) can be (word, characters), or (sentence, words)
     x: num_items x max_subitem_size x input_size
     x_mask: num_items x max_subitem_size (not used but put here to align with RNN format)
     return num_items x max_subitem_size x output_size
    '''
    def forward(self, x, x_mask):
        '''
         x_unsqueeze: num_items x 1 x max_subitem_size x input_size  
         x_conv: num_items x output_size x max_subitem_size
         x_output: num_items x max_subitem_size x output_size
        '''
        x = F.dropout(x, p = dropout_p, training = self.training)
        x_unsqueeze = x.unsqueeze(1) 
        x_conv = F.tanh(self.cnn(x_unsqueeze)).squeeze(3)
        x_output = torch.transpose(x_conv, 1, 2)
        return x_output


class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()
        self.MIN = -1e6

    '''
     (item, subitem) can be (word, characters), or (sentence, words)
     x: num_items x max_subitem_size x input_size
     x_mask: num_items x max_subitem_size
     return num_items x input_size
    '''
    def forward(self, x, x_mask):
        '''
         x_output: num_items x input_size x 1 --> num_items x input_size
        '''
        empty_mask = x_mask.eq(0).unsqueeze(2).expand_as(x)
        x_now = x.clone()
        x_now.data.masked_fill_(empty_mask.data, self.MIN)
        x_output = x_now.max(1)[0]
        x_output.data.masked_fill_(x_output.data.eq(self.MIN), 0)

        return x_output

class AveragePooling(nn.Module):
    def __init__(self):
        super(AveragePooling, self).__init__()

    '''
     (item, subitem) can be (word, characters), or (sentence, words)
     x: num_items x max_subitem_size x input_size
     x_mask: num_items x max_subitem_size
     return num_items x input_size
    '''
    def forward(self, x, x_mask):
        '''
         x_output: num_items x input_size x 1 --> num_items x input_size
        '''
        x_now = x.clone()
        empty_mask = x_mask.eq(0).unsqueeze(2).expand_as(x_now)
        x_now.data.masked_fill_(empty_mask.data, 0)
        x_sum = torch.sum(x_now, 1)
        # x_sum: num_items x input_size

        x_num = torch.sum(x_mask.eq(1).float(), 1).unsqueeze(1).expand_as(x_sum)
        # x_num: num_items x input_size

        x_num = torch.clamp(x_num, min = 1)

        return x_sum / x_num

class StackedBRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, rnn_type = nn.LSTM, concat_layers = False, bidirectional = True, add_feat=0, LN=False, batch_size=None, max_len=None):
        super(StackedBRNN, self).__init__()
        self.bidir_coef = 2 if bidirectional else 1
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.hidden_size = hidden_size
        self.rnns = nn.ModuleList()
        self.LN = LN
        if LN:
            self.ln = nn.LayerNorm([batch_size, max_len, self.bidir_coef * self.hidden_size])
        for i in range(num_layers):
            in_size = input_size if i == 0 else (self.bidir_coef * hidden_size + add_feat if i== 1 else self.bidir_coef * hidden_size)
            rnn = rnn_type(in_size, hidden_size, num_layers = 1, bidirectional = bidirectional, batch_first = True)
            self.rnns.append(rnn)
    
    @property
    def output_size(self):
        if self.concat_layers:
            return self.num_layers * self.bidir_coef * self.hidden_size
        else:
            return self.bidir_coef * self.hidden_size

    """
       Multi-layer bi-RNN
              
       Arguments:
           x (Float Tensor): a Float Tensor of size (batch * wordnum * input_dim).
           x_mask (Byte Tensor): a Byte Tensor of mask for the input tensor (batch * wordnum).
           x_additional (Byte Tensor): a Byte Tensor of mask for the additional input tensor (batch * wordnum * additional_dim).
           x_out (Float Tensor): a Float Tensor of size (batch * wordnum * output_size).
    """
    def forward(self, x, x_mask, return_list=False, x_additional = None, LN = None):
        hiddens = [x]
        for i in range(self.num_layers):
            rnn_input = hiddens[-1]
            if i == 1 and x_additional is not None:
                rnn_input = torch.cat((rnn_input, x_additional), 2)

            if dropout_p > 0:
                rnn_input = dropout(rnn_input, p=dropout_p, training = self.training)

            rnn_output = self.rnns[i](rnn_input)[0]
            if LN:
                rnn_output = F.layer_norm(rnn_output, rnn_output.size())
            assert torch.sum(torch.isnan(rnn_output)) == 0
            hiddens.append(rnn_output)

        if self.concat_layers:
            output = torch.cat(hiddens[1:], 2)
        else:
            output = hiddens[-1]

        if return_list:
            return output, hiddens[1:]
        else:
            return output

class AttentionScore(nn.Module):
    """
    correlation_func = 1, sij = x1^Tx2
    correlation_func = 2, sij = (Wx1)D(Wx2)
    correlation_func = 3, sij = Relu(Wx1)DRelu(Wx2)
    correlation_func = 4, sij = x1^TWx2
    correlation_func = 5, sij = Relu(Wx1)Relu(Wx2)
    """
    def __init__(self, input_size, hidden_size, correlation_func = 1, do_similarity = False):
        super(AttentionScore, self).__init__()
        self.correlation_func = correlation_func
        self.hidden_size = hidden_size
        
        if correlation_func == 2 or correlation_func == 3:
            self.linear = nn.Linear(input_size, hidden_size, bias = False)
            if do_similarity:
                self.diagonal = Parameter(torch.ones(1, 1, 1) / (hidden_size ** 0.5), requires_grad = False)
            else:
                self.diagonal = Parameter(torch.ones(1, 1, hidden_size), requires_grad = True)

        if correlation_func == 4:
            self.linear = nn.Linear(input_size, input_size, bias=False)

        if correlation_func == 5:
            self.linear = nn.Linear(input_size, hidden_size, bias = False)    
        
    def forward(self, x1, x2):
        '''
        Input:
        x1: batch x word_num1 x dim
        x2: batch x word_num2 x dim
        Output:
        scores: batch x word_num1 x word_num2
        '''
        x1 = dropout(x1, p = dropout_p, training = self.training)
        x2 = dropout(x2, p = dropout_p, training = self.training)

        x1_rep = x1
        x2_rep = x2
        batch = x1_rep.size(0)
        word_num1 = x1_rep.size(1)
        word_num2 = x2_rep.size(1)
        dim = x1_rep.size(2)
        if self.correlation_func == 2 or self.correlation_func == 3:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)  # Wx1
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)  # Wx2
            if self.correlation_func == 3:
                x1_rep = F.relu(x1_rep)
                x2_rep = F.relu(x2_rep)
            x1_rep = x1_rep * self.diagonal.expand_as(x1_rep) 
            # x1_rep is (Wx1)D or Relu(Wx1)D
            # x1_rep: batch x word_num1 x dim (corr=1) or hidden_size (corr=2,3)

        if self.correlation_func == 4:
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, dim)  # Wx2

        if self.correlation_func == 5:
            x1_rep = self.linear(x1_rep.contiguous().view(-1, dim)).view(batch, word_num1, self.hidden_size)  # Wx1
            x2_rep = self.linear(x2_rep.contiguous().view(-1, dim)).view(batch, word_num2, self.hidden_size)  # Wx2
            x1_rep = F.relu(x1_rep)
            x2_rep = F.relu(x2_rep)    
            
        scores = x1_rep.bmm(x2_rep.transpose(1, 2))
        return scores

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, correlation_func = 1, do_similarity = False):
        super(Attention, self).__init__()
        self.scoring = AttentionScore(input_size, hidden_size, correlation_func, do_similarity)
        # self.gelu = nn.GELU()

    def forward(self, x1, x2, x2_mask, x3 = None, drop_diagonal=False, return_score=False):
        '''
        For each word in x1, get its attended linear combination of x3 (if none, x2), 
         using scores calculated between x1 and x2.
        Input:
         x1: batch x word_num1 x dim
         x2: batch x word_num2 x dim
         x2_mask: batch x word_num2
         x3 (if not None) : batch x word_num2 x dim_3
        Output:
         attended: batch x word_num1 x dim_3
        '''
        batch = x1.size(0)
        word_num1 = x1.size(1)
        word_num2 = x2.size(1)

        if x3 is None:
            x3 = x2

        scores = self.scoring(x1, x2)

        # scores: batch x word_num1 x word_num2
        empty_mask = x2_mask.eq(0).unsqueeze(1).expand_as(scores)
        scores.data.masked_fill_(empty_mask.data, -float('inf'))

        if drop_diagonal:
            assert(scores.size(1) == scores.size(2))
            diag_mask = torch.diag(scores.data.new(scores.size(1)).zero_() + 1).byte().unsqueeze(0).expand_as(scores)
            scores.data.masked_fill_(diag_mask, -float('inf'))

        # softmax
        alpha_flat = F.softmax(scores.view(-1, x2.size(1)), dim = 1)
        alpha = alpha_flat.view(-1, x1.size(1), x2.size(1))
        # alpha: batch x word_num1 x word_num2

        attended = alpha.bmm(x3)
        # attended = self.gelu(attended)
        assert torch.sum(torch.isnan(attended)) == 0
        # attended: batch x word_num1 x dim_3
        if return_score:
            return attended, alpha.detach().cpu()
        else:
            return attended

def RNN_from_opt(input_size_, hidden_size_, num_layers=1, concat_rnn=False, add_feat=0, bidirectional=True, rnn_type=nn.LSTM, LN=False, batch_size=None, max_len=None):
    new_rnn = StackedBRNN(
        input_size=input_size_,
        hidden_size=hidden_size_,
        num_layers=num_layers,
        rnn_type=rnn_type,
        concat_layers=concat_rnn,
        bidirectional=bidirectional,
        add_feat=add_feat,
        LN = False,
        batch_size=batch_size,
        max_len=max_len
    )

    output_size = hidden_size_
    if bidirectional:
        output_size *= 2
    if concat_rnn:
        output_size *= num_layers

    return new_rnn, output_size   

# For summarizing a set of vectors into a single vector
class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        x = batch * len * hdim
        x_mask = batch * len
        """
        empty_mask = x_mask.eq(0).expand_as(x_mask)

        x = dropout(x, p=dropout_p, training=self.training)

        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(empty_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim = 1)
        return alpha

def generate_mask(new_data, dropout_p=0.0):
    new_data = (1-dropout_p) * (new_data.zero_() + 1)
    for i in range(new_data.size(0)):
        one = random.randint(0, new_data.size(1) - 1)
        new_data[i][one] = 1
    mask = Variable(1.0/(1 - dropout_p) * torch.bernoulli(new_data), requires_grad=False)
    return mask

# Get positional scores and scores for 'yes', 'no', 'unknown' cases
class GetFinalScores(nn.Module):
    def __init__(self, x_size, h_size, yesno ,no_answer, useES):
        super(GetFinalScores, self).__init__()
        self.no_answer = no_answer
        self.yesno = yesno
        self.useES = useES
        if no_answer:
            self.noanswer_linear = nn.Linear(h_size, x_size)
            self.noanswer_w = nn.Linear(x_size, 1, bias=True)
        if yesno:
            self.no_linear = nn.Linear(h_size, x_size)
            self.no_w = nn.Linear(x_size, 1, bias=True)
            self.yes_linear = nn.Linear(h_size, x_size)
            self.yes_w = nn.Linear(x_size, 1, bias=True)
            self.no_read_linear = nn.Linear(h_size, x_size)
            self.no_read_w = nn.Linear(x_size, 1, bias=True)
            #self.no_read = LinearTransform(h_size, 1)
        self.attn = BilinearSeqAttn(x_size, h_size)
        self.rnn = nn.GRUCell(x_size, h_size)
        self.attn2 = BilinearSeqAttn(x_size, h_size)

    def forward(self, x, h0, x_mask, ES_len, mask_flag=None):
        """
        x = batch * len * x_size
        h0 = batch * h_size
        x_mask = batch * len
        score_es = batch * max_es_len
        score_ocr = batch * (max_ocr_num-max_es_len)
        x_es = batch * max_es_len * x_size
        x_ocr = batch * (max_ocr_num-max_es_len) * x_size
        """
        if self.useES:
            x_es = x[:,:ES_len]
            x_es_mask = x_mask[:,:ES_len]
            x_ocr = x[:,ES_len:]
            x_ocr_mask = x_mask[:,ES_len:]
            score_ocr = self.attn(x_ocr, h0, x_ocr_mask, mask_flag=mask_flag)
            #print(score_ocr.shape)
            #print(x.shape)
            #print(F.softmax(score_ocr, dim = 1).shape)
            #print(F.softmax(score_ocr, dim = 1).unsqueeze(1).shape)
            # score_s_return = F.softmax(score_s, dim = 1)
            
            ptr_net_in = torch.bmm(F.softmax(score_ocr, dim = 1).unsqueeze(1), x_ocr).squeeze(1)
            ptr_net_in = dropout(ptr_net_in, p=dropout_p, training=self.training)
            h1 = self.rnn(ptr_net_in, h0)
            # h1 same size as h0

            score_es = self.attn2(x_es, h0, x_es_mask, mask_flag=mask_flag)
            score_s = torch.cat([score_es, score_ocr], dim=-1)
        else:
            score_s = self.attn(x, h0, x_mask, mask_flag=mask_flag)

        if self.yesno:
            h0 = dropout(h0, p=dropout_p, training=self.training)
            score_no = self.get_single_score(x, h0, x_mask, self.no_linear, self.no_w)
            score_yes = self.get_single_score(x, h0, x_mask, self.yes_linear, self.yes_w)
            score_noread = self.get_single_score(x, h0, x_mask, self.no_read_linear, self.no_read_w)
            #score_noread = self.no_read(h0)
            score_s = torch.cat([score_noread, score_yes, score_no, score_s], dim=-1)

        if self.no_answer:
            h0 = dropout(h0, p=dropout_p, training=self.training)
            score_noanswer = self.get_single_score(x, h0, x_mask, self.noanswer_linear, self.noanswer_w)
            score_s = torch.cat([score_s, score_noanswer], dim=-1)
        # return score_s, score_e, score_no, score_yes, score_noanswer
        score_s = F.softmax(score_s, dim=-1)
        return score_s
    
    def get_single_score(self, x, h, x_mask, linear, w):
        Wh = linear(h)  #batch * x_size
        xWh = x.bmm(Wh.unsqueeze(2)).squeeze(2) #batch * len

        empty_mask = x_mask.eq(0).expand_as(x_mask)
        xWh.data.masked_fill_(empty_mask.data, -float('inf'))

        attn_x = torch.bmm(F.softmax(xWh, dim = 1).unsqueeze(1), x) # batch * 1 * x_size
        single_score = w(attn_x).squeeze(2) # batch * 1
        assert torch.sum(torch.isnan(single_score)) == 0

        return single_score

# For attending the span in document from the query
class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = x_i'Wy for x_i in X.
    """
    def __init__(self, x_size, y_size, identity=False):
        super(BilinearSeqAttn, self).__init__()
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None          

    def forward(self, x, y, x_mask, mask_flag=True):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        empty_mask = x_mask.eq(0).expand_as(x_mask)

        x = dropout(x, p=dropout_p, training=self.training)
        y = dropout(y, p=dropout_p, training=self.training)

        Wy = self.linear(y) if self.linear is not None else y  # batch * h1
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)  # batch * len
        # print('{:7.3} {:7.3} {:7.3} {:7.3}'.format(xWy.max().item(), xWy.min().item(), xWy.mean().item(), xWy.std().item()), end='')
        # xWy.data.masked_fill_(empty_mask.data, 0)
        # print('### {:7.3} {:7.3} {:7.3} {:7.3}'.format(xWy.max().item(), xWy.min().item(), xWy.mean().item(), xWy.std().item()))
        assert torch.sum(torch.isnan(xWy)) == 0
        if mask_flag:
            xWy.data.masked_fill_(empty_mask.data, -float('inf'))
            #xWy = F.softmax(xWy, dim=-1)
            # xWy.data.masked_fill_(empty_mask.data, 0)
        assert torch.sum(torch.isnan(xWy)) == 0
        return xWy

# History-of-Word Multi-layer inter-attention
class DeepAttention(nn.Module):
    def __init__(self, opt, abstr_list_cnt, deep_att_hidden_size_per_abstr, correlation_func=1, word_hidden_size=None):
        super(DeepAttention, self).__init__()

        word_hidden_size = opt['embedding_dim'] if word_hidden_size is None else word_hidden_size
        abstr_hidden_size = opt['hidden_size'] * 2
        if 'no_DeepAttention' in opt:
            att_size = 0
            rnn_input_size = abstr_hidden_size * abstr_list_cnt
        else:
            att_size = abstr_hidden_size * abstr_list_cnt + word_hidden_size
            self.int_attn_list = nn.ModuleList()
            for i in range(abstr_list_cnt+1):
                self.int_attn_list.append(Attention(att_size, deep_att_hidden_size_per_abstr, correlation_func = correlation_func))
            rnn_input_size = abstr_hidden_size * abstr_list_cnt * 2 + (opt['highlvl_hidden_size'] * 2)
        
        self.att_size = att_size
        self.rnn_input_size = rnn_input_size
        self.rnn, self.output_size = RNN_from_opt(rnn_input_size, opt['highlvl_hidden_size'], num_layers=1)

        self.opt = opt

    def forward(self, x1_word, x1_abstr, x2_word, x2_abstr, x1_mask, x2_mask, return_bef_rnn=False, return_score=False):
        """
        x1_word, x2_word, x1_abstr, x2_abstr are list of 3D tensors.
        3D tensor: batch_size * length * hidden_size
        """
        if 'no_DeepAttention' not in self.opt:
            x1_att = torch.cat(x1_word + x1_abstr, 2)
            x2_att = torch.cat(x2_word + x2_abstr[:-1], 2)
            x1 = torch.cat(x1_abstr, 2)

            x2_list = x2_abstr
            if return_score:
                att_score = []
            for i in range(len(x2_list)):
                if return_score:
                    attn_hiddens, _att_score = self.int_attn_list[i](x1_att, x2_att, x2_mask, x3=x2_list[i], return_score=True)
                    att_score.append(_att_score)
                else:
                    attn_hiddens = self.int_attn_list[i](x1_att, x2_att, x2_mask, x3=x2_list[i])
                x1 = torch.cat((x1, attn_hiddens), 2)
        else:
            x1 = torch.cat(x1_abstr, 2)

        x1_hiddens = self.rnn(x1, x1_mask)
        if return_bef_rnn and return_score:
            return x1_hiddens, x1, att_score
        elif return_bef_rnn:
            return x1_hiddens, x1
        elif return_score:
            return x1_hiddens, att_score
        else:
            return x1_hiddens

# bmm: batch matrix multiplication
# unsqueeze: add singleton dimension
# squeeze: remove singleton dimension
def weighted_avg(x, weights): # used in lego_reader.py
    """ 
        x = batch * len * d
        weights = batch * len
    """
    return weights.unsqueeze(1).bmm(x).squeeze(1)

class LinearTransform(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearTransform, self).__init__()
        self.lc = weight_norm(
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=False), dim=None
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.lc(x)
