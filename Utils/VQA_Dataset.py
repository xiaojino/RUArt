from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os, logging, torch, json
from Models.Bert.tokenization import BertTokenizer
import Utils.eval_func as eval_func
from Utils.phoc import build_phoc
from Utils.eval_func import note_stvqa, note_textvqa
log = logging.getLogger(__name__)

# global opt
# opt = {}

class VQA_Dataset(Dataset):
    def __init__(self, data, opt, mode='train', image_features=None, fixed_answers_entry=None):
        self.opt = opt
        self.data = []
        self.mode = mode
        assert mode in ['train', 'dev', 'test']
        error_samples = []
        for datum in data:
            if len(datum['annotated_question']['word']) == 0:
                error_samples.append(datum['question_id'])
                continue
            if mode != 'test' and len(datum['orign_answers']) == 0:
                error_samples.append(datum['question_id'])
                continue
            self.data.append(datum)
        # global self.opt
        log.info('Remove {} samples for empty question or answers: {}'.format(len(error_samples), error_samples))
        self.opt = opt
        self.set_dataset()
        if 'DEBUG' in self.opt:
            self.debug = True
        else:
            self.debug = False
        self.debug_dataset()
        self.img_features_cache = {}
        self.image_features = image_features
        self.fixed_answers_entry = fixed_answers_entry

        if 'ES_ocr' in self.opt:
            self.ocr_name_list = [self.opt['ES_ocr']] + self.ocr_name_list
            self.es_ocr_len = int(self.opt['ES_ocr_len'])
            self.es_sort_way = self.opt['ES_sort_way']
        log.info('Using OCR from: {}'.format(self.ocr_name_list))
        log.info('Using OD from: {}'.format(self.od_name_list))
        
        if 'BERT' in self.opt:
            if 'BERT_LARGE' in self.opt:
                log.debug('Using BERT Large model')
                tokenizer_file = os.path.join(self.opt['datadir'], self.opt['BERT_large_tokenizer_file'])
                log.debug('Loading tokenizer from {}'.format(tokenizer_file))
                self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_file)
            else:
                log.debug('Using BERT base model')
                tokenizer_file = os.path.join(self.opt['datadir'], self.opt['BERT_tokenizer_file'])
                log.debug('Loading tokenizer from {}'.format(tokenizer_file))
                self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_file)
    def set_dataset(self):
        self.ocr_name_list = self.opt['ocr_name_list'].split(',')
        self.od_name_list = self.opt['od_name_list'].split(',')
        self.q_embedding = self.opt['q_embedding'].split(',')
        self.ocr_embedding = self.opt['ocr_embedding'].split(',')
        self.score_name = self.opt['score_name']
        

        self.max_ocr_num = self.opt['max_ocr_num']
        self.max_od_num = self.opt['max_od_num']
        self.max_ocr_len = self.opt['max_ocr_len']
        self.max_od_len = self.opt['max_od_len']
        self.max_q_len = self.opt['max_q_len']
    def debug_dataset(self):
        if not self.debug:
            self.q_output = None
            self.ocr_output = None
            self.od_output = None
        else:
            self.q_output = {
                'glove_len': {},
                'bert_len': {},
                'bert_only_len': {},
                'ocr_num': {},
                'od_num': {}
            }
            self.ocr_output = {
                'glove_len': {},
                'bert_len': {},
                'bert_only_len': {}
            }
            self.od_output = {
                'glove_len': {},
                'bert_len': {},
                'bert_only_len': {}
            }
    def print_debug(self, dtn):
        def save(ojb, file_path):
            with open(file_path, 'w') as wf:
                json.dump(ojb, wf, indent=2)
        
        save(self.q_output, dtn + '_q_output.json')
        save(self.ocr_output, dtn + '_ocr_output.json')
        save(self.od_output, dtn + '_od_output.json')

    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        # print(self.opt)
        ocr_list = self.get_list_from_datum(datum, self.ocr_name_list, od_ocr='ocr', remove_same='remove_same' in self.opt)
        od_list = self.get_list_from_datum(datum, self.od_name_list, od_ocr='od', remove_same='remove_same' in self.opt)
        datum['annotated_question']['original'] = datum['question'].lower()
        
        q = self.get_item_embedding(datum['annotated_question'], self.q_embedding, self.q_output)

        if self.debug:
            ocr_len = len(ocr_list)
            od_len = len(od_list)
            if ocr_len in self.q_output['ocr_num']:
                self.q_output['ocr_num'][ocr_len] += 1
            else:
                self.q_output['ocr_num'][ocr_len] = 1
            if od_len in self.q_output['od_num']:
                self.q_output['od_num'][od_len] += 1
            else:
                self.q_output['od_num'][od_len] = 1
        ocr_list = ocr_list[:self.max_ocr_num]
        od_list = od_list[:self.max_od_num]
        ocr = self.get_list_embedding(ocr_list, self.ocr_embedding, self.ocr_output)
        od = self.get_list_embedding(od_list, self.ocr_embedding, self.od_output)

        answers = datum['orign_answers'] if 'orign_answers' in datum else None
        gt = self.get_label(ocr_list, q_id=datum['question_id'], answers=answers)

        extra_info = {
            'q_id': datum['question_id'],
            'answers': datum['orign_answers'] if 'orign_answers' in datum else None,
            'ocr_list': [t['original'] for t in ocr_list],
            'image_path': datum['filename'],
        }
        if 'img_feature' in self.opt:
            q['img_features'], q['img_spatials'] = self.get_image_feature(datum['filename'], datum['question_id'])
        batch = {
            'q': q,
            'ocr': ocr, 
            'od': od,
            'gt': gt,
            'extra_info': extra_info
        }

        return batch
    def get_image_feature(self, image_path, q_id):
        if self.image_features != None:
            idx = self.image_features['img_id2idx'][q_id]
            img_fea = self.image_features['img_features'][q_id]
            img_fea = img_fea.unsqueeze(0)
            bbox = self.image_features['img_spatials'][q_id]
            img_spa = torch.FloatTensor(1, bbox.size(0), 8).fill_(0)
            img_spa[0, :, 0] = bbox[:, 0]
            img_spa[0, :, 1] = bbox[:, 1]
            img_spa[0, :, 2] = bbox[:, 2]
            img_spa[0, :, 3] = bbox[:, 1]
            img_spa[0, :, 4] = bbox[:, 2]
            img_spa[0, :, 5] = bbox[:, 3]
            img_spa[0, :, 6] = bbox[:, 0]
            img_spa[0, :, 7] = bbox[:, 3]
            assert img_fea.size() == (1, 36, 2048)
            assert img_spa.size() == (1, 36, 8)
            return img_fea, img_spa



        if image_path in self.img_features_cache:
            return self.img_features_cache[image_path]
        image_path_without_ext = ''.join(image_path.split('.')[:-1])
        img_fea_folder = self.opt['img_fea_folder']
        if self.mode != 'test':
            img_fea_folder = os.path.join(img_fea_folder, 'train')
        else:
            img_fea_folder = os.path.join(img_fea_folder, 'test')
        img_fea_path = os.path.join(img_fea_folder, image_path_without_ext+'.npy')
        assert os.path.exists(img_fea_path)
        img_fea_np = np.load(img_fea_path)
        img_fea_tensor = torch.from_numpy(img_fea_np).unsqueeze(0)

        img_fea_path = os.path.join(img_fea_folder, image_path_without_ext+'_info.npy')
        assert os.path.exists(img_fea_path)
        img_info = np.load(img_fea_path, allow_pickle=True).item()
        bbox = torch.from_numpy(img_info['bbox'])
        bbox[:, 0] = bbox[:, 0] / img_info['image_width']
        bbox[:, 2] = bbox[:, 2] / img_info['image_width']
        bbox[:, 1] = bbox[:, 1] / img_info['image_height']
        bbox[:, 3] = bbox[:, 3] / img_info['image_height']

        img_spa = torch.FloatTensor(1, bbox.size(0), 8).fill_(0)
        img_spa[0, :, 0] = bbox[:, 0]
        img_spa[0, :, 1] = bbox[:, 1]
        img_spa[0, :, 2] = bbox[:, 2]
        img_spa[0, :, 3] = bbox[:, 1]
        img_spa[0, :, 4] = bbox[:, 2]
        img_spa[0, :, 5] = bbox[:, 3]
        img_spa[0, :, 6] = bbox[:, 0]
        img_spa[0, :, 7] = bbox[:, 3]
        self.img_features_cache[image_path] = (img_fea_tensor, img_spa)
        return self.img_features_cache[image_path]
    
    
    
    def get_label(self, ocr_list, q_id=None, answers=None):
        textvqa = False
        if self.score_name not in ocr_list[0]:
            return None, None
        gt_ynu_num = 0
        gt = [t[self.score_name] for t in ocr_list]
        if 'label_yesno' in self.opt:
            if self.score_name == 'ANLS':
                gt_yes = note_stvqa(answers, 'yes')
                gt_no = note_stvqa(answers, 'no')
                gt_noread = note_stvqa(answers, 'answering does not require reading text in the image')
            elif self.score_name == 'ACC':
                gt_yes = note_textvqa(answers, 'yes')
                gt_no = note_textvqa(answers, 'no')
                gt_noread = note_textvqa(answers, 'answering does not require reading text in the image')
            gt = [gt_noread,gt_yes,gt_no]+ gt
            gt_ynu_num = 3

        if self.fixed_answers_entry != None:
            fixed_ans_gt = self.fixed_answers_entry['fixed_answers_label'][q_id]
            fixend_ans_len = self.fixed_answers_entry['fixed_answers_len']
            if len(fixed_ans_gt) == 2:
                self.fixed_gt_sample[fixed_ans_gt[0]] = fixed_ans_gt[1]
                textvqa = True
            else:
                assert len(fixed_ans_gt) == 2
        else:
            fixed_ans_gt = []
            fixend_ans_len = 0

        if 'fixed_answers' in self.opt:
            gt = fixed_ans_gt + gt

        gt_max = -1
        gt_max_idx = -1
        for idx, t in enumerate(gt):
            if t > gt_max:
                gt_max = t
                gt_max_idx = idx
        #gt_max = max(gt_max,gt_yes,gt_no,gt_noread)


        if self.opt['lable_way'] == 'lable_all':
            gt = gt
        elif self.opt['lable_way'] == 'lable_all_with_threshold':
            gt = [t if t >= self.opt['score_threshold'] else 0 for t in gt]
        elif self.opt['lable_way'] == 'lable_one_offical':
            if self.score_name == 'ANLS':
                gt = [t if i == gt_max_idx and gt_max >= 0.5 else 0 for i, t in enumerate(gt)]
            elif self.score_name == 'ACC':
                gt = [t if i == gt_max_idx and gt_max >= 0.3 else 0 for i, t in enumerate(gt)]
        elif self.opt['lable_way'] == 'lable_one':
            gt = [t if i == gt_max_idx else 0 for i, t in enumerate(gt)]
        else:
            log.error('lable_way is wrong')
            assert False


        #gt_tensor = torch.FloatTensor(1, fixend_ans_len).fill_(0)
        if 'fixed_answers' in self.opt:
            gt_tensor = torch.FloatTensor(1, gt_ynu_num + self.max_ocr_num + fixend_ans_len).fill_(0)
        else:
            gt_tensor = torch.FloatTensor(1, gt_ynu_num + self.max_ocr_num).fill_(0)

        _len = len(gt)
        gt_tensor[0, :_len] = torch.FloatTensor(gt)
        #if 'lable_yesno' in self.opt:
        #    gt_y = torch.FloatTensor(1, 1).fill_(gt_yes)
        #    gt_n = torch.FloatTensor(1, 1).fill_(gt_no)
        #    gt_tensor = torch.cat([gt_tensor, gt_y], dim=1)
        #    gt_tensor = torch.cat([gt_tensor, gt_n], dim=1)

        if 'label_no_answer' in self.opt:
            if gt_max < 0.1:
                no_answer = torch.FloatTensor(1, 1).fill_(1.0)
            else:
                no_answer = torch.FloatTensor(1, 1).fill_(0.0)
            gt_tensor = torch.cat([gt_tensor, no_answer], dim=1)

        if textvqa:
            self.fixed_gt_sample[fixed_ans_gt[0]] = 0
        return gt_tensor


    def get_list_from_datum(self, datum, name_list, od_ocr='ocr', remove_same=False):
        '''
        TODO: 
            Remove same OCRs between different kinds of OCR model results by IoU, Now remove these by OCR tokens
        MODIFY:
            add <OCR>/<OD> token for empty OCR results. Dont use them when finally predicting answers

        '''
        assert od_ocr in ['od', 'ocr']
        vocab_set = {}
        res = []
        for name in name_list:
            if 'ES_ocr' in self.opt and name == self.opt['ES_ocr']:
                if self.es_sort_way == 'frequency':
                    datum[name].sort(key=lambda x: x['cnt'], reverse=True)
                elif self.es_sort_way == 'relevance':
                    datum[name].sort(key=lambda x: x['idx'])
                else:
                    log.error('es_sort_way is wrong')
                    assert False
                datum[name] = datum[name][:self.es_ocr_len]
            for item in datum[name]:
                if od_ocr == 'od':
                    item['word'] = item['object']
                if len(item['word']['word']) == 0:
                    continue
                k = item['original'].lower()
                item['original'] = k
                if 'ES_ocr' in self.opt and name == self.opt['ES_ocr']:
                    res.append(item)
                    continue
                if remove_same and (k in vocab_set):
                    continue
                vocab_set[k] = 1
                res.append(item)
        if od_ocr == 'od':
            if len(res) >= self.max_od_num-1:
                res = res[:self.max_od_num-1]
        else:
            if len(res) >= self.max_ocr_num-1:
                res = res[:self.max_ocr_num-1]
        end_word = {
            'word': ['<OCR>' if od_ocr == 'ocr' else '<OD>'],
            'wordid': [3 if od_ocr == 'ocr' else 4],
            'pos_id': [0],
            'ent_id': [0],
        }
        end = {
            'word': end_word,
            'pos': [0 for _ in range(8)],
            'original': '<OCR>' if od_ocr == 'ocr' else '<OD>',
            'ANLS': 0.0,
            'ACC': 0.0,
        }
        res.append(end)
        assert len(res) > 0
        return res
    


    def get_item_embedding(self, item, embedding_list, output, original=None):
        res = {}
        if 'fasttext' in embedding_list:
            res['fasttext'] = item['wordid']
        if 'phoc' in embedding_list:
            res['phoc'] = item['wordid']
        if 'glove' in embedding_list:
            _len = len(item['wordid'])
            if self.debug:
                if _len in output['glove_len']:
                    output['glove_len'][_len] += 1
                else:
                    output['glove_len'][_len] = 1
            res['glove'] = item['wordid']
        if 'pos' in embedding_list:
            res['pos'] = item['pos_id']
        if 'ent' in embedding_list:
            res['ent'] = item['ent_id']

        if 'bert' in self.q_embedding:
            x_bert, x_bert_offsets = self.bertify(item['word'])
            if self.debug:
                x_bert_len = len(x_bert)
                if x_bert_len in output['bert_len']:
                    output['bert_len'][x_bert_len] += 1
                else:
                    output['bert_len'][x_bert_len] = 1
            res['bert'] = x_bert
            res['bert_offsets'] = x_bert_offsets

        if 'bert_only' in self.q_embedding:
            if 'original' in item:
                x_bert, x_bert_offsets = self.bertify(item['original'])
            else:
                assert original != None
                x_bert, x_bert_offsets = self.bertify(original)
            if self.debug:
                x_bert_len = len(x_bert)
                if x_bert_len in output['bert_only_len']:
                    output['bert_only_len'][x_bert_len] += 1
                else:
                    output['bert_only_len'][x_bert_len] = 1
            res['bert_only'] = x_bert
        return res


    def get_list_embedding(self, item_list, embedding_list, output):
        res = []
        for idx, item in enumerate(item_list):
            if 'object' in item:
                tmp = self.get_item_embedding(item['object'], embedding_list, output, original=item['original'])
            else:
                tmp = self.get_item_embedding(item['word'], embedding_list, output, original=item['original'])
                #assert len(tmp['glove']) > 0
                # if len(tmp['glove']) > 1 and idx >= 10:
                #     print('multi tokens {}: {}'.format(item['original'], item['word']['word']))
            tmp['position'] = item['pos']
            res.append(tmp)
        return res

    def bertify(self, words):
        if self.bert_tokenizer is None:
            return None

        bpe = ['[CLS]']
        x_bert_offsets = []
        if isinstance(words, list):
            for word in words:
                now = self.bert_tokenizer.tokenize(word)
                x_bert_offsets.append([len(bpe), len(bpe) + len(now)])
                bpe.extend(now)
            if len(words) == 0:
                x_bert_offsets = [1,1]
        elif isinstance(words, str):
            bpe = bpe + self.bert_tokenizer.tokenize(words)
        else:
            log.error('BERT tokenizer is wrong')
            assert False
        bpe.append('[SEP]')

        x_bert = self.bert_tokenizer.convert_tokens_to_ids(bpe)
        return x_bert, x_bert_offsets


class VQA_collate():
    def __init__(self, opt):
        self.opt = opt
        # if 'ModelParallel' in self.opt:
        #     self.bert_cuda = 'cuda:{}'.format(self.opt['ModelParallel'][-1])
        #     self.main_cuda = 'cuda:{}'.format(self.opt['ModelParallel'][0])
        # else:
        #     self.bert_cuda = None
        #     self.main_cuda = None
    def VQA_collate_fun(self, batch):
        q_list = [t['q'] for t in batch]
        ocr_list = [t['ocr'] for t in batch]
        od_list = [t['od'] for t in batch]
        gt_list = [t['gt'] for t in batch]
        extra_info_list = [t['extra_info'] for t in batch]

        max_ocr_num = self.opt['max_ocr_num']
        max_od_num = self.opt['max_od_num']
        max_ocr_len = self.opt['max_ocr_len']
        max_od_len = self.opt['max_od_len']
        max_ocr_bert_len = self.opt['max_ocr_bert_len']
        max_od_bert_len = self.opt['max_od_bert_len']
        max_q_len = self.opt['max_q_len']
        max_q_bert_len = self.opt['max_q_bert_len']
        
        q_list = self.que_collate(q_list, max_q_len, max_q_bert_len)
        ocr_list = self.item_collate(ocr_list, max_ocr_len, max_ocr_bert_len, max_ocr_num)
        od_list = self.item_collate(od_list, max_od_len, max_od_bert_len, max_od_num)
        gt_list = self.gt_collate(gt_list)

        return q_list, ocr_list, od_list, gt_list, extra_info_list

    def gt_collate(self, gt_list):
        gt = torch.cat(gt_list, dim=0)
        return gt


    def item_collate(self, item_list, max_len, max_bert_len, max_num):
        res = {}
        batch_size = len(item_list)
        for k in item_list[0][0].keys():
            if 'offset' in k:
                emb = [_t[k] for t in item_list for _t in t]
                res[k] = emb
            elif k == 'position':
                emb = torch.FloatTensor(batch_size, max_num, 8).fill_(0)
                for idx, item in enumerate(item_list):
                    _len = len(item)
                    emb[idx][:_len] = torch.FloatTensor([t[k] for t in item])
                res[k] = emb
            else:
                for idx, item in enumerate(item_list):
                    if k in ['bert', 'bert_only']:
                        emb = torch.LongTensor(len(item), max_bert_len).fill_(0)
                    else:
                        emb = torch.LongTensor(len(item), max_len).fill_(0)
                    for _i, _item in enumerate(item):
                        _len = len(_item[k])
                        emb[_i][:_len] = torch.LongTensor(_item[k])
                    if k in res:
                        res[k] = torch.cat([res[k], emb], dim=0)
                    else:
                        res[k] = emb
        keys = [k for k in res.keys()]
        for k in keys:
            if 'offset' in k:
                continue
            # if k in ['bert', 'bert_only'] and bert_cuda != None:
            #     res[k] = res[k].cuda(bert_cuda)
            # else:
            #     res[k] = res[k].cuda()
            if k in ['glove', 'fasttext', 'phoc', 'bert', 'bert_only']:
                res[k+'_mask'] = ~ res[k].eq(0)
        res['num_cnt'] = [len(item) for item in item_list]
        if 'FastText' in self.opt:
            res['len_cnt'] = [[len(_t['fasttext']) for _t in t] for t in item_list]
        else:
            res['len_cnt'] = [[len(_t['glove']) for _t in t] for t in item_list]
        return res

    def que_collate(self, q_list, max_len, max_bert_len):
        batch_size = len(q_list)
        res = {}
        for k in q_list[0].keys():
            if k in ['img_features', 'img_spatials']:
                emb = torch.cat([t[k] for t in q_list], dim=0)
            elif 'offset' in k:
                emb = [t[k] for t in q_list]
            else:
                if k in ['bert', 'bert_only']:
                    emb = torch.LongTensor(batch_size, max_bert_len).fill_(0)
                else:
                    emb = torch.LongTensor(batch_size, max_len).fill_(0)
                for idx, item in enumerate(q_list):
                    _len = len(item[k])
                    emb[idx][:_len] = torch.LongTensor(item[k])
                # if 'bert' in k and bert_cuda != None:
                #     emb = emb.cuda(bert_cuda)
                # else:
                #     emb = emb.cuda()
                if k in ['fasttext','glove', 'phoc', 'bert', 'bert_only']:
                    res[k+'_mask'] = ~ emb.eq(0)
            res[k] = emb
        return res
            