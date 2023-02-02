import numpy as np

def char_tokenizer(s):
    '''
    문자열을 문자들의 리스트로 만든다. 이 때 앞에 공란이 오는 문자에는 밑 줄을 붙인다.

    Arguments
    ---------
    s : str
        입력 문자열, 보통 중간에 공란이 있는 문장이다.
    '''
    result = []
    flag = False
    for c in s:
        if c == ' ':
            flag = True
            continue
        if flag == True:
            c = '_' + c
            flag = False
        result.append(c)
    return result

def list_tokenizer(l):
    '''
    단어들의 리스트를 문자들의 리스트로 만든다. 이 때 단어의 맨 앞의 문자에는 밑줄(_)을 붙인다.

    Arguments
    ---------
    l : list
        단어(문자열)들의 리스트

    Retrun
    ---------
    result : list
        단어(문자열)들의 리스트
    '''
    result = []
    flag = False
    for word in l:
        result.append('_' + word[0])
        result.extend(list(word[1:]))

    result[0] = result[0].replace('_', '')
    return result

def get_swords(ref , hyp):
    '''
    두 개의 문자열을 비교하여 띄어쓰기를 맞추어 준다.
    ref를 기준으로 hyp의 띄어쓰기를 맞춘다.

    Arguments
    ---------
    ref : str
        입력 문자열, 보통 중간에 공란이 있는 문장이다. 이 문자열을 기준으로 hyp의 띄어쓰기를 맞춘다.
    hyp : str
        입력 문자열, 보통 중간에 공란이 있는 문장이다.
    '''
    # print(f'ref : {ref}')
    # print(f'hyp : {hyp}')
    refs = char_tokenizer(ref)
    hyps = char_tokenizer(hyp)
    ref_nospace = ref.replace(' ', '')
    hyp_nospace = hyp.replace(' ', '')
    # print(f'refs : {refs}')
    # print(f'hyps : {hyps}')
    rlen = len(refs)
    hlen = len(hyps)
    scores =  np.zeros((hlen+1, rlen+1), dtype=np.int32)

    # initialize, 공란을 무시하고 음절의 거리 매트릭스 만들기
    for r in range(rlen+1):
        scores[0, r] = r
    for h in range(1, hlen+1):
        scores[h, 0] = scores[h-1, 0] + 1
        for r in range(1, rlen+1):
            sub_or_cor = scores[h-1, r-1] + (0 if ref_nospace[r-1] == hyp_nospace[h-1] else 1)
            insert = scores[h-1, r] + 1
            delete = scores[h, r-1] + 1
            scores[h, r] = min(sub_or_cor, insert, delete)

    # traceback and compute alignment
    h, r = hlen, rlen
    ref_norm, hyp_norm = [], []

    while r > 0 or h > 0:
        if h == 0:
            last_r = r - 1
        elif r == 0:
            last_h = h - 1
            last_r = r
        else:
            sub_or_cor = scores[h-1, r-1] + (0 if ref_nospace[r-1] == hyp_nospace[h-1] else 1)
            insert = scores[h-1, r] + 1
            delete = scores[h, r-1] + 1

            if sub_or_cor < min(insert, delete):
                last_h, last_r = h - 1, r - 1
            else:
                last_h, last_r = (h-1, r) if insert < delete else (h, r-1)

            c_hyp = hyps[last_h] if last_h == h-1 else ''
            c_ref = refs[last_r] if last_r == r-1 else ''
        h, r = last_h, last_r

        # do word-spacing normalization
        if c_hyp.replace('_', '') == c_ref.replace('_', ''):
            c_hyp = c_ref

        ref_norm.append(c_ref)
        hyp_norm.append(c_hyp)

    # ref_norm[::-1], hyp_norm[::-1]
    shyp = ''.join(map(str, hyp_norm[::-1])).replace('_', ' ')
    return shyp

def space_normalize_lists(ref_list, hyp_list):
    '''
    두 개의 단어들의 리스트를 비교하여 띄어쓰기를 맞추어 준다.(space normalize)
    ref_list를 기준으로 hyp_list의 띄어쓰기를 맞춘다.

    Arguments
    ---------
    ref_list : list
        단어(문자열)들의 리스트
        이 리스트를 기준으로 hyp_list의 띄어쓰기를 맞춘다.
    hyp_list : list
        단어(문자열)들의 리스트

    Retrun
    ---------
    result : list
        ref_list에 띄어쓰기를 맞춘 hyp_list의 리스트

    ex)
    ---------
    ref_list = ['나는', '어제', '양념치킨을', '먹었다']
    hyp_list = ['나는어제', '치킨을먹었다']
    output : ['나는', '어제치킨을', '먹었다']
    '''

    if not (any(ref_list) and any(hyp_list)):
        return hyp_list

    while '' in hyp_list:
        # print(f'hyp_list 1 : {hyp_list}')
        hyp_list.remove('')
        # print(f'hyp_list 2 : {hyp_list}')

    ref_nospace = ''.join(ref_list)
    hyp_nospace = ''.join(hyp_list)
    # print(f'ref_nospace : {ref_nospace}')
    # print(f'hyp_nospace : {hyp_nospace}')

    refs = list_tokenizer(ref_list)
    hyps = list_tokenizer(hyp_list)
    # print(f'refs : {refs}')
    # print(f'hyps : {hyps}')

    rlen = len(refs)
    hlen = len(hyps)
    scores =  np.zeros((hlen+1, rlen+1), dtype=np.int32)

    # initialize, 공란을 무시하고 음절의 거리 매트릭스 만들기
    for r in range(rlen+1):
        scores[0, r] = r
    for h in range(1, hlen+1):
        scores[h, 0] = scores[h-1, 0] + 1
        for r in range(1, rlen+1):
            sub_or_cor = scores[h-1, r-1] + (0 if ref_nospace[r-1] == hyp_nospace[h-1] else 1)
            insert = scores[h-1, r] + 1
            delete = scores[h, r-1] + 1
            scores[h, r] = min(sub_or_cor, insert, delete)
    
    # print(f'lev scores--------------------------')
    # print(scores)
    
    # traceback and compute alignment
    h, r = hlen, rlen
    ref_norm, hyp_norm = [], []

    while r > 0 or h > 0:
        if h == 0:
            last_r = r - 1
        elif r == 0:
            last_h = h - 1
            last_r = r
        else:
            sub_or_cor = scores[h-1, r-1] + (0 if ref_nospace[r-1] == hyp_nospace[h-1] else 1)
            insert = scores[h-1, r] + 1
            delete = scores[h, r-1] + 1

            if sub_or_cor < min(insert, delete):
                last_h, last_r = h - 1, r - 1
            else:
                last_h, last_r = (h-1, r) if insert < delete else (h, r-1)

            c_hyp = hyps[last_h] if last_h == h-1 else ''
            c_ref = refs[last_r] if last_r == r-1 else ''

        h, r = last_h, last_r
        # do word-spacing normalization
        if c_hyp.replace('_', '') == c_ref.replace('_', ''):
            c_hyp = c_ref

        ref_norm.append(c_ref)
        hyp_norm.append(c_hyp)
    # print(f'after while : r={r}, h={h}')

    while '' in ref_norm:
        ref_norm.remove('')
    while '' in hyp_norm:
        hyp_norm.remove('')
    # print(f'norm--------------------------')
    # print(f'ref_norm : {ref_norm[::-1]}')
    # print(f'hyp_norm : {hyp_norm[::-1]}')
    # print(f'hyp_norm[::-1] : {hyp_norm[::-1]}')
    shyp = ''.join(hyp_norm[::-1]).split('_')
    # print(f'shyp : {shyp}')
    return shyp