# RUArt
This repository is the implementation of "RUArt: A Novel Text-Centered Solution for
Text-Based Visual Question Answering".

This repository is based on and inspired by @microsoft's [work](https://github.com/microsoft/SDNet) . We sincerely thank for their sharing of the codes.

## Directory structure
Download pytorch [bert-base-uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) model from huggingface, then extract it into #root/source, where #root is the code root folder.

The directory structure is as follows:
- RUArt
  - conf~
    - model
    (Downlod the [pretrained RUArt model](https://drive.google.com/open?id=1T-JViYKu9iNdcKHjRTGxYFAA9vunmWAz) to this folder.)
  - Models
  - Utils
  - source
    - data
      (Download the [preprocessed training and test files](https://drive.google.com/open?id=1UNEFmatxclidwQAXOpCjDbuV6FVbXv7H) and extract it into this folder.)
    - bert-base-uncased
      - bert_config.json
      - pytorch_model.bin
      - vocab.txt
  - conf
  - main_test.py
  - readme.md
## Requirements
### pip install
```bash
pip3 install -r requiresments.txt
```
## Inference
```bash
cd #root
python main_test.py
```
The result of ST-VQA task3 test will be saved in conf~/model/submission.json
