# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os, logging
log = logging.getLogger()

class BaseTrainer():
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = False
        if self.opt['cuda'] == True:
            self.use_cuda = True
            print('Using Cuda\n') 
        else:
            self.use_cuda = False
            print('Using CPU\n')

        self.is_official = 'OFFICIAL' in self.opt
        # Make sure raw text feature files are ready
        self.use_spacy = 'SPACY_FEATURE' in self.opt
        self.opt['logFile'] = 'log.txt'

        opt['FEATURE_FOLDER'] = './source/data/' + self.opt['source_dir'] + '/'
        opt['FEATURE_FOLDER'] = os.path.join(opt['datadir'], opt['FEATURE_FOLDER'])

    def log(self, s):
        # In official case, the program does not output logs
        if self.isTrain == False:
            print(s)
            return
        if self.is_official:
            return

        with open(os.path.join(self.saveFolder, self.opt['logFile']), 'a') as f:
            f.write(s + '\n')
        print(s)
    def myLog(self):
        i = 0
        while os.path.exists(self.saveFolder+'/logging_'+str(i)+'.log'):
            i += 1
        log_path = os.path.join(self.saveFolder, 'logging_' + str(i)+'.log')
        file_handler = logging.FileHandler(log_path, 'w')
        formatter = logging.Formatter("%(asctime)s - %(filename)s - %(levelname)s: %(message)s")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
        # log.info('add log file handler')
        # print('ok')

    def getSaveFolder(self):
        if self.isTrain:
            runid = 1
            while True:
                saveFolder = os.path.join(self.opt['datadir'], 'conf~', 'run_' + str(runid))
                if not os.path.exists(saveFolder):
                    self.saveFolder = saveFolder
                    os.makedirs(self.saveFolder)
                    print('Saving logs, model and evaluation in ' + self.saveFolder)
                    break
                runid = runid + 1
        else:
            p = '/'.join(self.opt['MODEL_PATH'].split('/')[:2])
            self.saveFolder = os.path.join(self.opt['datadir'], p)
        self.myLog()
  
    # save copy of conf file 
    def saveConf(self):
        with open(self.opt['confFile'], encoding='utf-8') as f:
            with open(os.path.join(self.saveFolder, 'conf_copy'), 'w', encoding='utf-8') as fw:
                for line in f:
                    fw.write(line)

    def train(self): 
        pass
 
    def load(self):
        pass
