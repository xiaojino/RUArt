# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from Models.Bert.modeling import BertModel
import logging
log = logging.getLogger(__name__)

'''
    BERT    
'''
class Bert(nn.Module):
    def __init__(self, opt):
        super(Bert, self).__init__()
        print('Loading BERT model...')
        self.BERT_MAX_LEN = 512
        self.linear_combine = 'BERT_LINEAR_COMBINE' in opt
        if 'BERT_MAX_BatchSize' in opt:
            self.BERT_MAX_BS = opt['BERT_MAX_BatchSize']
            log.info('split bert embedding with BatchSize {}'.format(self.BERT_MAX_BS))
        else :
            self.BERT_MAX_BS = None
        
        if 'BERT_LARGE' in opt:
            # print('Using BERT Large model')
            model_file = os.path.join(opt['datadir'], opt['BERT_large_model_file'])
            # print('Loading BERT model from', model_file)
            self.bert_model = BertModel.from_pretrained(model_file)
            #self.bert_model = BertModel.from_pretrained('bert-large-uncased')        
            self.bert_dim = 1024
            self.bert_layer = 24
        else:
            # print('Using BERT base model')
            model_file = os.path.join(opt['datadir'], opt['BERT_model_file'])
            # print('Loading BERT model from', model_file)
            self.bert_model = BertModel.from_pretrained(model_file)
            #self.bert_model = BertModel.from_pretrained('bert-base-cased')        
            self.bert_dim = 768
            self.bert_layer = 12
        self.bert_model.cuda()
        self.bert_model.eval()

        print('Finished loading')

    '''
        Input:
              x_bert: batch * max_bert_sent_len (ids)
              x_bert_mask: batch * max_bert_sent_len (0/1)
              x_bert_offset: batch * max_real_word_num * 2
              x_mask: batch * max_real_word_num
            Output:
              embedding: batch * max_real_word_num * bert_dim
    '''
    def forward(self, x_bert, x_bert_mask, x_bert_offset, x_mask, device=None):
        # while x_bert.size(0) < 5000:
        #     x_bert = torch.cat([x_bert, x_bert], dim=0)
        #     x_bert_mask = torch.cat([x_bert_mask, x_bert_mask], dim=0)
        #     x_bert_offset = torch.cat([x_bert_offset, x_bert_offset], dim=0)
        #     x_mask = torch.cat([x_mask, x_mask], dim=0)
        batch_size = x_bert.size(0)
        # log.debug('x_bert: '+str(x_bert.size()))
        # split_flag = True
        if self.BERT_MAX_BS != None:
            st = 0
            res = []
            log.debug('x_bert:{} x_bert_mask:{} x_mask:{}'.format(x_bert.size(), x_bert_mask.size(), x_mask.size()))
            while st < batch_size:
                ed = min(st+self.BERT_MAX_BS, batch_size)
                if self.linear_combine:
                    res.append(self.combine_forward(x_bert[st:ed], x_bert_mask[st:ed], x_bert_offset[st:ed], x_mask[st:ed], device=device))
                else:
                    res.append(self.original_forward(x_bert[st:ed], x_bert_mask[st:ed], x_bert_offset[st:ed], x_mask[st:ed], device=device))
                st += self.BERT_MAX_BS
            if self.linear_combine:
                out = []
                for i in range(len(res[0])):
                    out.append(torch.cat([res[t][i] for t in range(len(res))], dim=0))
                # log.debug('output[0]:{}'.format(out[0].size()))
                return out
            else:
                out = torch.cat(res, dim=0)
                # log.debug('output:{}'.format(out.size()))
                return out
        else:
            if self.linear_combine:
                return self.combine_forward(x_bert, x_bert_mask, x_bert_offset, x_mask, device=device)
            else:
                return self.original_forward(x_bert, x_bert_mask, x_bert_offset, x_mask, device=device)
        # assert False
    def original_forward(self, x_bert, x_bert_mask, x_bert_offset, x_mask, device=None):
        last_layers = []
        bert_sent_len = x_bert.shape[1]
        p = 0
        while p < bert_sent_len:
            all_encoder_layers, _ = self.bert_model(x_bert[:, p:(p + self.BERT_MAX_LEN)], token_type_ids=None, attention_mask=x_bert_mask[:, p:(p + self.BERT_MAX_LEN)]) # bert_layer * batch * max_bert_sent_len * bert_dim
            last_layers.append(all_encoder_layers[-1]) # batch * up_to_512 * bert_dim
            p += self.BERT_MAX_LEN

        bert_embedding = torch.cat(last_layers, 1)
        if x_bert_offset == None:
            if device != None:
                return bert_embedding.to(device)
            else:
                return bert_embedding.cuda()
        
        batch_size = x_mask.shape[0]
        max_word_num = x_mask.shape[1]
        output = Variable(torch.zeros(batch_size, max_word_num, self.bert_dim))
        for i in range(batch_size):
            for j in range(max_word_num):
                if x_mask[i, j] == 0:
                    continue
                st = x_bert_offset[i][j][0]    
                ed = x_bert_offset[i][j][1]
                # we can also try using st only, ed only
                if st + 1 == ed: # including st==ed
                    output[i, j, :] = bert_embedding[i, st, :]
                else:    
                    subword_ebd_sum = torch.sum(bert_embedding[i, st:ed, :], dim = 0)
                    if st < ed:
                        output[i, j, :] = subword_ebd_sum / float(ed - st) # dim 0 is st:ed
        if device != None:
            output = output.to(device)
        else:
            output = output.cuda()        
        return output

    def combine_forward(self, x_bert, x_bert_mask, x_bert_offset, x_mask, device=None):
        all_layers = []

        bert_sent_len = x_bert.shape[1]
        p = 0
        while p < bert_sent_len:
            all_encoder_layers, _ = self.bert_model(x_bert[:, p:(p + self.BERT_MAX_LEN)], token_type_ids=None, attention_mask=x_bert_mask[:, p:(p + self.BERT_MAX_LEN)]) # bert_layer * batch * max_bert_sent_len * bert_dim
            all_layers.append(torch.cat(all_encoder_layers, dim = 2))  # batch * up_to_512 * (bert_dim * layer)
            p += self.BERT_MAX_LEN

        bert_embedding = torch.cat(all_layers, dim = 1) # batch * up_to_512 * (bert_dim * layer)
        if x_bert_offset is None:
            outputs = []
            for i in range(self.bert_layer):
                now = bert_embedding[:, :, (i * self.bert_dim) : ((i + 1) * self.bert_dim)]
                now = now.cuda()
                outputs.append(now)

            return outputs
        batch_size = x_mask.shape[0]
        max_word_num = x_mask.shape[1]
        tot_dim = bert_embedding.shape[2]
        output = Variable(torch.zeros(batch_size, max_word_num, tot_dim))
        for i in range(batch_size):
            for j in range(max_word_num):
                if x_mask[i, j] == 0:
                    continue
                st = x_bert_offset[i][j][0]    
                ed = x_bert_offset[i][j][1]
                # we can also try using st only, ed only
                if st + 1 == ed: # including st==ed
                    output[i, j, :] = bert_embedding[i, st, :]
                else:    
                    subword_ebd_sum = torch.sum(bert_embedding[i, st:ed, :], dim = 0)
                    if st < ed:
                        output[i, j, :] = subword_ebd_sum / float(ed - st) # dim 0 is st:ed

        outputs = []
        for i in range(self.bert_layer):
            now = output[:, :, (i * self.bert_dim) : ((i + 1) * self.bert_dim)]
            if device != None:
                now = now.to(device)
            else:
                now = now.cuda()
            outputs.append(now)

        return outputs
