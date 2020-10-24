def stvqa_score(str1, str2):
    str1 = str1.lower()
    str2 = str2.lower()
    if max(len(str2), len(str1)) == 0:
        return 1
    def Levenshtein_Distance(str1, str2):
        """
        计算字符串 str1 和 str2 的编辑距离
        :param str1
        :param str2
        :return:
        """
        matrix = [[ i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

        for i in range(1, len(str1)+1):
            for j in range(1, len(str2)+1):
                if(str1[i-1] == str2[j-1]):
                    d = 0
                else:
                    d = 1
                
                matrix[i][j] = min(matrix[i-1][j]+1, matrix[i][j-1]+1, matrix[i-1][j-1]+d)

        return matrix[len(str1)][len(str2)]
    # print(Levenshtein_Distance("abc", "bd"))
    ld = Levenshtein_Distance(str1, str2)
    score = 1 - ld / max(len(str2), len(str1))
    return score
def note_stvqa(gt_list, word):
    # gt_list = [t.lower() for t in gt_list]
    s = -1
    for gt in gt_list:
        _s = stvqa_score(gt, word)
        s = max(_s, s)
    return s

def stvqa_lable(gt_list, ocr_list):
    # gt_list = [t.lower() for t in gt_list]
    ocr_list = [t['word'] for t in ocr_list]
    all_none = True
    lable_score = -1
    lable_idx = -1
    for gt in gt_list:
        if len(gt) == 0:
            continue
        else:
            all_none = False
        gt_ls = -1
        gt_li = -1
        for ocr_idx, ocr in enumerate(ocr_list):
            ocr_ls = stvqa_score(gt, ocr)
            if ocr_ls > gt_ls:
                gt_ls = ocr_ls
                gt_li = ocr_idx
        if gt_ls > lable_score:
            lable_score = gt_ls
            lable_idx = gt_li
    if all_none:
        return False
    return lable_idx, lable_score
# print(stvqa_score('abc', 'bd'))
def note_textvqa(gt_list, word):
    gt_list = [t.lower() for t in gt_list]
    cnt = 0
    for gt in gt_list:
        if gt == word:
            cnt += 1
    return cnt / 10.0



def textvqa_lable(gt_list, ocr_list):
    gt_list = [t.lower() for t in gt_list]
    ocr_list = [t['word'] for t in ocr_list]
    lable_score = -1
    lable_idx = -1
    # assert len(gt_list) == 10
    for ocr_idx, ocr in enumerate(ocr_list):
        ocr_cnt = 0
        for gt in gt_list:
            if ocr == gt:
                ocr_cnt += 1
        ocr_ls = ocr_cnt / 10.0
        if ocr_ls > lable_score:
            lable_score = ocr_ls
            lable_idx = ocr_idx
    # score = min(lable_score * 10 / 3.0, 1.0)
    return lable_idx, lable_score