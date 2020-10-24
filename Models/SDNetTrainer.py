from datetime import datetime
import json
import numpy as np
import os
import msgpack
import random
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Utils.CoQAPreprocess import CoQAPreprocess
from Models.Layers import MaxPooling, set_dropout_prob
from Models.SDNet import SDNet
from Models.BaseTrainer import BaseTrainer
from Utils.CoQAUtils import BatchGen, AverageMeter, gen_upper_triangle, score
import pickle, h5py
from tqdm import tqdm
from Utils.VQA_Dataset import VQA_Dataset, VQA_collate
from Utils.VQA_Sampler import VQA_Sampler
from torch.utils.data import DataLoader
from Utils.eval_func import note_stvqa, note_textvqa
from Utils.phoc import build_phoc
import logging
log = logging.getLogger(__name__)
 
class SDNetTrainer(BaseTrainer):
    def __init__(self, opt):
        super(SDNetTrainer, self).__init__(opt)
        print('SDNet Model Trainer')
        self.opt = opt
        set_dropout_prob(0.0 if not 'DROPOUT' in opt else float(opt['DROPOUT']))
        self.seed = int(opt['SEED'])
        self.data_prefix = 'vqa-'
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.preproc = CoQAPreprocess(self.opt)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.seed)
        self.batch_size = self.opt['batch_size']
        if 'ModelParallel' in self.opt:
            self.bert_cuda = 'cuda:{}'.format(self.opt['ModelParallel'][-1])
            self.main_cuda = 'cuda:{}'.format(self.opt['ModelParallel'][0])
        else:
            self.bert_cuda = None
            self.main_cuda = None
        self.load_fixed_answers()

    def train(self): 
        self.isTrain = True
        self.getSaveFolder()
        self.saveConf()
        self.vocab, self.char_vocab, vocab_embedding = self.preproc.load_data()
        log.info('-----------------------------------------------')
        log.info("Initializing model...")
        self.setup_model(vocab_embedding)
        self.reture_att_score = 'att_score' in self.opt
        
        if 'RESUME' in self.opt:
            model_path = os.path.join(self.opt['datadir'], self.opt['MODEL_PATH'])
            self.load_model(model_path)
        VQA_collate_fun = VQA_collate(self.opt).VQA_collate_fun
        self.VQA_collate_fun = VQA_collate_fun
        if 'DEBUG' in self.opt:
            for dtn in ['train', 'val', 'test']:
                with open(os.path.join(self.opt['FEATURE_FOLDER'], dtn+'-preprocessed.msgpack'), 'rb') as f:
                    test_data = msgpack.load(f, encoding='utf8')
                test_data = VQA_Dataset(test_data['data'], self.opt)
                test_sampler = VQA_Sampler(test_data, self.opt['max_batch_num'], self.batch_size, False)
                test_loader = DataLoader(test_data, batch_sampler=test_sampler, collate_fn=VQA_collate_fun)
                print(dtn)
                for batch_i, batch in tqdm(enumerate(test_loader)):
                    a = 1
                del test_data, test_sampler, test_loader
                # test_data.print_debug(dtn)
            assert False

        log.info('Loading train json...')
        with open(os.path.join(self.opt['FEATURE_FOLDER'], 'train-preprocessed.msgpack'), 'rb') as f:
            train_data = msgpack.load(f, encoding='utf8')
        with open(os.path.join(self.opt['FEATURE_FOLDER'], 'val-preprocessed.msgpack'), 'rb') as f:
            val_data = msgpack.load(f, encoding='utf8')
        log.info('Dataset has been loaded')

        self.best_ANLS = -1
        self.best_ACC = -1
        self.best_ANLS_batch = self.best_ACC_batch = -1

        batch_st = self.opt['batch_st'] if 'batch_st' in self.opt else 0
        num_worker = 0
        if 'num_worker' in self.opt:
            num_worker = self.opt['num_worker']
        self.load_image_features()

        train_data = VQA_Dataset(train_data['data'], self.opt, image_features=self.img_features, fixed_answers_entry=self.fixed_answers_entry)
        train_sampler = VQA_Sampler(train_data, self.opt['max_batch_num'], self.batch_size, True, batch_st=batch_st, epoch=self.opt['epoch'] if 'epoch' in self.opt else None)
        train_loader = DataLoader(train_data, batch_sampler=train_sampler, collate_fn=VQA_collate_fun, num_workers=num_worker)

        val_data = VQA_Dataset(val_data['data'], self.opt, image_features=self.img_features, fixed_answers_entry=self.fixed_answers_entry)
        

        train_st = datetime.now()
        val_interval = datetime.now() - datetime.now()
        for batch_i, batch in enumerate(train_loader):
            batch_i += batch_st
            batch = self.ToCUDA(batch)

            val_st = datetime.now()
            if batch_i % 1500 == 0:
                self.evaluate(val_data, batch_i)
            val_interval += datetime.now() - val_st

            _loss = self.update(batch, batch_i)
            if batch_i % 30 == 0:
                log.info('updates[{0:6}] train loss[{1:8.5f} / {2:8.5f}] remaining[{3}]'.format(
                    self.updates, 
                    self.train_loss.avg,
                    _loss,
                    str((datetime.now() - train_st - val_interval) / (batch_i + 1) * (len(train_loader) - batch_st - batch_i - 1)).split('.')[0]))
            torch.cuda.empty_cache()
        self.evaluate(val_data, batch_i)
        self.evaluate(train_data, batch_i, mode='train')
        log.info('Training over')

    def evaluate(self, val_data, batch_i, mode='dev'):
        assert mode in ['train', 'dev', 'test']
        val_len = len(val_data)
        val_sampler = VQA_Sampler(val_data, self.opt['max_batch_num'], self.batch_size, False)
        val_loader = DataLoader(val_data, batch_sampler=val_sampler, collate_fn=self.VQA_collate_fun)
        with torch.no_grad():
            loss = ANLS = ACC = 0
            res = []
            save_res = []
            for val_i, val_batch in enumerate(val_loader):
                val_batch = self.ToCUDA(val_batch)
                _loss, _ANLS, _ACC, _res, _save_res = self.predict(val_batch)
                loss += _loss
                ANLS += _ANLS
                ACC += _ACC
                res.extend(_res)
                save_res.extend(_save_res)
            loss = loss / len(val_loader)
            ANLS = ANLS / val_len
            ACC = ACC / val_len
        if mode == 'test':
            log.info('{} test samples are predicted'.format(val_len))
            end = val_len % self.batch_size
            if end != 0:
                end = self.batch_size - end
                res = res[:-end]
            log.info('{} predictions are saved'.format(len(res)))
            
            submission_path = os.path.join(self.saveFolder, 'submission.json')
            with open(submission_path, 'w') as wf:
                json.dump(res, wf, indent=2)
                wf.close()
            log.info('submission is saved in {}'.format(submission_path))
            return
        if mode == 'dev':
            with open(os.path.join(self.saveFolder, 'save_res_last.json'), 'w') as wf:
                json.dump(save_res, wf, indent=2)
                wf.close()
            if ANLS > self.best_ANLS:
                self.best_ANLS = ANLS
                self.best_ANLS_batch = batch_i
                model_file = os.path.join(self.saveFolder, 'ANLS_best_model.pt')
                self.save_for_predict(model_file)
            if ACC > self.best_ACC:
                self.best_ACC = ACC
                self.best_ACC_batch = batch_i
                model_file = os.path.join(self.saveFolder, 'ACC_best_model.pt')
                self.save_for_predict(model_file)
        log.info('Dataset: {} Batch: {:7} ANLS: {:.3f} Best ANLS: {:.3f} Batch: {} ACC: {:.3f} Best ACC:{:.3} Batch:{}'.format(mode, batch_i, ANLS, self.best_ANLS, self.best_ANLS_batch, ACC, self.best_ACC, self.best_ACC_batch))

    def load_image_features(self):
        if 'img_feature' in self.opt:
            log.info('Loading image features...')
            img_feature_folder = os.path.dirname(os.path.dirname(os.path.dirname(self.opt['FEATURE_FOLDER'])))
            img_feature_folder = os.path.join(img_feature_folder, 'image_features')
            train_img_id2idx = pickle.load(open(os.path.join(img_feature_folder, 'train36_imgid2idx.pkl'), 'rb'))
            val_img_id2idx = pickle.load(open(os.path.join(img_feature_folder, 'val36_imgid2idx.pkl'), 'rb'))
            with h5py.File(os.path.join(img_feature_folder, 'train36.hdf5'), 'r') as hf:
                train_img_features = torch.tensor(hf.get('image_features'))
                train_img_spatials = torch.tensor(hf.get('spatial_features'))
                train_num = train_img_features.size(0)
            with h5py.File(os.path.join(img_feature_folder, 'val36.hdf5'), 'r') as hf:
                val_img_features = torch.tensor(hf.get('image_features'))
                val_img_spatials = torch.tensor(hf.get('spatial_features'))

            img_id2idx = train_img_id2idx
            
            for k,v in val_img_id2idx.items():
                assert k not in img_id2idx
                img_id2idx[k] = v + train_num
            img_features = torch.cat([train_img_features, val_img_features], dim=0)
            img_spatials = torch.cat([train_img_spatials, val_img_spatials], dim=0)
            self.img_features = {
                'img_features': img_features,
                'img_spatials': img_spatials,
                'img_id2idx': img_id2idx
            }
            log.info('Image features have been loaded')
        else:
            self.img_features = None    
    def ToCUDA(self, batch):
        # q_list, ocr_list, od_list, gt_list, extra_info = batch
        res = []
        for idx, item in enumerate(batch):
            if idx < 3:
                for k in item.keys():
                    flag = True
                    if k in ['bert', 'bert_only', 'bert_mask', 'bert_only_mask']:
                        if self.bert_cuda != None:
                            item[k] = item[k].to(self.bert_cuda)
                        else:
                            item[k] = item[k].cuda()
                    elif k in ['fasttext', 'fasttext_mask', 'phoc', 'phoc_mask', 'glove', 'glove_mask', 'ent', 'pos', 'position', 'img_features', 'img_spatials']:
                        item[k] = item[k].cuda()
                    else:
                        flag = False
                    if flag:
                        if torch.sum(torch.isnan(item[k])) > 0:
                            print('input nan')
            elif idx == 3:
                item = item.cuda()
            res.append(item)
        return res
    def predict_for_test(self): 
        self.reture_att_score = 'att_score' in self.opt
        self.isTrain = False
        self.getSaveFolder()
        self.vocab, self.char_vocab, vocab_embedding = self.preproc.load_data()
        print('-----------------------------------------------')
        print("Initializing model...")
        self.setup_model(vocab_embedding)
        print('Loading {}...'.format(os.path.join(self.opt['FEATURE_FOLDER'], 'test-preprocessed.msgpack')))
        with open(os.path.join(self.opt['FEATURE_FOLDER'], 'test-preprocessed.msgpack'), 'rb') as f:
            test_data = msgpack.load(f, encoding='utf8')
        assert 'RESUME' in self.opt
        model_path = os.path.join(self.opt['datadir'], self.opt['MODEL_PATH'])
        self.load_model(model_path)
        self.VQA_collate_fun = VQA_collate(self.opt).VQA_collate_fun
        self.load_image_features()
        self.num_worker = 0
        if 'num_worker' in self.opt:
            self.num_worker = self.opt['num_worker']
        test_data = VQA_Dataset(test_data['data'], self.opt, image_features=self.img_features, fixed_answers_entry=self.fixed_answers_entry, mode='test')
        self.evaluate(test_data, 0, 'test')
    
    def load_fixed_answers(self):
        if 'fixed_answers' in self.opt:
            fixed_answers_set = {}
            fixed_answers = []
            # os.path.dirname(os.path.dirname(os.path.dirname(self.opt['FEATURE_FOLDER'])))
            fix_ans_label_path = os.path.join(self.opt['fixed_answers_folder'], 'TRAIN_VAL_fixed_answers_label.msgpack')
            fix_ans_path = os.path.join(self.opt['fixed_answers_folder'], 'fixed_answers_4000.txt')

            log.info('Loading Fixed Answers....')
            with open(fix_ans_path, 'r') as wf:
                for line in wf.readlines():
                    line = line.strip()
                    line = line.lower()
                    assert line not in fixed_answers_set
                    fixed_answers_set[line] = len(fixed_answers)
                    fixed_answers.append(line)
                wf.close()
            with open(fix_ans_label_path, 'rb') as wf:
                fixed_answers_label = msgpack.load(wf, encoding='utf8')
            fixed_answers_len = len(fixed_answers)
            if 'phoc' in self.opt['ocr_embedding']:
                fixed_answers_phoc = [build_phoc(t) for t in fixed_answers]
            else:
                fixed_answers_phoc = None
            self.fixed_answers_entry = {
                'fixed_answers_set': fixed_answers_set,
                'fixed_answers_len': fixed_answers_len,
                'fixed_answers_phoc': fixed_answers_phoc,
                'fixed_answers': fixed_answers,
                'fixed_answers_label': fixed_answers_label,
            }
            self.fixed_answers_len = fixed_answers_len
            log.info('Fixed Answers have been loaded')
        else:
            self.fixed_answers_entry = None
            self.fixed_answers_len = 0

    def setup_model(self, vocab_embedding):
        self.train_loss = AverageMeter()
        self.network = SDNet(self.opt, vocab_embedding)
        # self.network = nn.DataParallel(SDNet(self.opt, vocab_embedding))
        if self.use_cuda:
            log.info('Putting model into GPU')
            if 'ModelParallel' in self.opt:
                bert_cuda = 'cuda:{}'.format(self.opt['ModelParallel'][-1])
                main_cuda = 'cuda:{}'.format(self.opt['ModelParallel'][0])
                self.network.to(main_cuda)
                self.network.Bert.to(bert_cuda)
                log.info('SDNet on {}, Bert on {}'.format(main_cuda, bert_cuda))
            else:
                self.network.cuda()

        parameters = [p for p in self.network.parameters() if p.requires_grad]

        if self.opt['optimizer'] == 'ADAM':
            self.optimizer = optim.Adamax(parameters, weight_decay=0.5, lr=1e-3)
        elif self.opt['optimizer'] == '#':
            self.optimizer = optim.Adamax(parameters, lr=self.opt['lr'] if 'lr' in self.opt else 2e-3)
        elif self.opt['optimizer'] == 'ADAM2':
            self.optimizer = optim.Adam(parameters, lr=self.opt['lr'] if 'lr' in self.opt else 1e-3)
        elif self.opt['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(parameters, lr=self.opt['lr'])
        else:
            log.error('optimizer is wrong')
            assert False
        log.info('Optimizer: {}, lr={:.3}'.format(self.opt['optimizer'], self.opt['lr'] if 'lr' in self.opt else -1))

        self.updates = 0
        self.epoch_start = 0
        if self.opt['loss'] in ['BCE', 'BCE_D1']:
            self.loss_func = self.instance_bce_with_logits
        elif self.opt['loss'] == 'CE':
            self.loss_func = F.cross_entropy
        else:
            log.error('loss parameter is error')
            assert False

    def update(self, batch, batch_i):
        # Train mode
        self.network.train()
        self.network.drop_emb = True
        
        
        q_list, ocr_list, od_list, targets, extra_info = batch
        scores, _ = self.network(q_list, ocr_list, od_list)

        nan_flag = False
        if torch.sum(torch.isnan(scores)) > 0:
            log.info('scores nan')
            nan_flag = True
        if self.opt['loss'] == 'CE':
            targets = torch.nonzero(targets)[:,1]
        if torch.sum(torch.isnan(targets)) > 0:
            log.info('targets nan')
            nan_flag = True
        if torch.sum(torch.isnan(scores)) > 0:
            log.info('scores nan')
            nan_flag = True
        #print(str(scores))
        #print(str(targets))
        loss = self.loss_func(scores, targets)

        if torch.sum(torch.isnan(loss)) > 0:
            log.info('loss nan')
            nan_flag = True
        if nan_flag:
            assert False

        if 'DEBUG_SDT' in self.opt:
            print(loss.item(), [t['q_id'] for t in extra_info])
        self.train_loss.update(loss.item(), 1)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.opt['grad_clipping'])
        self.optimizer.step()
        self.updates += 1
        if 'TUNE_PARTIAL' in self.opt:
            if 'FastText' in self.opt:
                self.network.fast_embed.weight.data[self.opt['tune_partial']:] = self.network.fixed_embedding_fast
            if 'GLOVE' in self.opt:
                self.network.glove_embed.weight.data[self.opt['tune_partial']:] = self.network.fixed_embedding_glove
        # log.debug('updata successes')
        # torch.cuda.empty_cache()
        return loss.item()

    def predict(self, batch, all_ans=False):
        self.network.eval()
        self.network.drop_emb = False

        # Run forward
        q_list, ocr_list, od_list, gt_list, extra_info = batch
        scores,_ = self.network(q_list, ocr_list, od_list)

        if gt_list != None:
            loss = self.loss_func(scores, gt_list)
        else:
            loss = 0
        prob = scores.detach().cpu()

        res = []
        save_res = []
        ANLS = ACC = 0
        batch_size = scores.size(0)
        if 'label_yesno' in self.opt:
            yesno_num = 3
        else:
            yesno_num = 0

        # print(prob.size())
        for i in range(batch_size):
            # assert extra_info[i]['ocr_list'][-1] == '<OCR>'
            _, ids = torch.sort(prob[i, :], descending=True)
            for idx in ids:
                if 'label_no_answer' in self.opt:
                    if idx == ids.size(0)-1:
                        break
                if idx == len(extra_info[i]['ocr_list']) - 1:
                    continue
                if idx < (self.fixed_answers_len + yesno_num + ocr_list['num_cnt'][i]):
                    break

            if idx < self.fixed_answers_len:
                answer = self.fixed_answers_entry['fixed_answers'][idx]
            elif idx < self.fixed_answers_len + yesno_num:
                if idx < self.fixed_answers_len + 1:
                    answer = 'answering does not require reading text in the image'
                elif idx < self.fixed_answers_len + 2:
                    answer = 'yes'
                elif idx < self.fixed_answers_len + 3:
                    answer = 'no'
            elif idx < (self.fixed_answers_len + yesno_num + ocr_list['num_cnt'][i]):
                answer = extra_info[i]['ocr_list'][idx-self.fixed_answers_len-yesno_num]
            else:
                answer = 'unanswerable'
            # print(answer, idx, ocr_list['num_cnt'][i])
            res.append({
                'question_id': extra_info[i]['q_id'],
                'answer': answer
            })
            save_res.append({
                'question_id': extra_info[i]['q_id'],
                'prediction': answer,
                'answers': extra_info[i]['answers'],
                'score': prob[i, idx].item(),
                'idx': idx.item(),
                'ids_len': len(ids),
                'ocr_list': extra_info[i]['ocr_list']
            })
            if extra_info[i]['answers'] != None:
                _anls = note_stvqa(extra_info[i]['answers'], answer)
                _acc = note_textvqa(extra_info[i]['answers'], answer)
                if len(extra_info[i]['answers']) == 10:
                    ACC += min(_acc * 10 / 3.0, 1)
                else:
                    ACC += min(_acc * 10, 1)
                ANLS += _anls if _anls >= 0.5 else 0

        assert len(res) == batch_size
        return loss.item(), ANLS, ACC, res, save_res # list of strings, list of floats, list of jsons

    def load_model(self, model_path):
        print('Loading model from', model_path)
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        new_state = set(self.network.state_dict().keys())
        for k in list(state_dict['network'].keys()):
            if k not in new_state:
                del state_dict['network'][k]
        for k, v in list(self.network.state_dict().items()):
            if k not in state_dict['network']:
                state_dict['network'][k] = v
        self.network.load_state_dict(state_dict['network'])

        print('Loading finished', model_path)        

    def save(self, filename, epoch, prev_filename):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates # how many updates
            },
            'train_loss': {
                'val': self.train_loss.val,
                'avg': self.train_loss.avg,
                'sum': self.train_loss.sum,
                'count': self.train_loss.count
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            log.info('model saved to {}'.format(filename))
            if os.path.exists(prev_filename):
                os.remove(prev_filename)
        except BaseException:
            log.info('[ WARN: Saving failed... continuing anyway. ]')

    def save_for_predict(self, filename):
        network_state = dict([(k, v) for k, v in self.network.state_dict().items() if k[0:4] != 'CoVe' and k[0:4] != 'ELMo' and k[0:9] != 'AllenELMo' and k[0:4] != 'Bert'])

        if 'eval_embed.weight' in network_state:
            del network_state['eval_embed.weight']
        if 'fixed_embedding_fast' in network_state:
            del network_state['fixed_embedding_fast']
        if 'fixed_embedding_glove' in network_state: #glove
            del network_state['fixed_embedding_glove']
        params = {
            'state_dict': {'network': network_state},
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            log.info('model saved to {}'.format(filename))
        except BaseException:
            log.info('[ WARN: Saving failed... continuing anyway. ]')
    def instance_bce_with_logits(self, logits, labels):
        # labels = labels.float()
        # labels = F.softmax(labels, dim=-1)
        assert logits.dim() == 2

        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        if self.opt['loss'] == 'BCE_D1':
            loss *= labels.size(1)
        return loss
