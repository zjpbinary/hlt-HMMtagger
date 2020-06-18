if __name__ == '__main__':
    #train.conll文件预处理
    with open("data/train.conll", 'r', encoding='UTF-8') as f1:
        text = f1.readlines()
    text = [elem.strip().split('\t') for elem in text]
    sent = []
    sentset = []
    wordset = []
    tagset = []
    for e in text:
        if e == ['']:
            sentset.append(sent)
        else:
            sent.append((e[1], e[3]))
            wordset.append(e[1])
            tagset.append(e[3])
    wordset = set(wordset)
    tagset = set(tagset)

    #实现word2index，文本数字化
    word2index = {}
    tag2index = {}
    for i, word in enumerate(wordset):
        word2index[word] = i
    for i, tag in enumerate(tagset):
        tag2index[tag] = i+1
    with open("data/word2index", 'w', encoding='UTF-8') as f2:
        for k, v in word2index.items():
            f2.write(k)
            f2.write(' ')
            f2.write(str(v))
            f2.write('\n')
    with open('data/tag2index', 'w', encoding='UTF-8') as f3:
        for k, v in tag2index.items():
            f3.write(k)
            f3.write(' ')
            f3.write(str(v))
            f3.write('\n')
    sentset = [[(word2index[pair[0]], tag2index[pair[1]]) for pair in sent] for sent in sentset]

    #统计，建立概率分布q
    q = [[0 for _ in range(len(tagset)+2)] for _ in range(len(tagset)+2)]
    for sent in sentset:
        q[0][sent[0][1]] += 1
        q[sent[-1][1]][-1] += 1
        for i in range(len(sent)-1):
            q[sent[i][1]][sent[i+1][1]] += 1
    tagdict = dict()
    for sent in sentset:
        for pair in sent:
            tagdict[pair[1]] = tagdict.get(pair[1], 0) + 1
    tagnum = sum(tagdict.values())
    a1 = 0.7
    a2 = 0.3
    for i in range(len(q)-1):
        s = sum(q[i])
        q[i] = [(tagdict.get(j, 0)/tagnum*a2)+(q[i][j]/s*a1) for j in range(len(q[i]))]

    #统计，建立概率分布e,考虑未登录词
    e = [[1 for _ in range(len(tagset))] for i in range(len(wordset)+1)]#+1考虑了未登录词的问题，采用+1平滑
    for sent in sentset:
        for pair in sent:
            e[pair[0]][pair[1]-1] += 1 #注意e的词性序号比tag的索引小1
    for i in range(len(e[0])):
        s = sum([e[j][i] for j in range(len(e))])
        for j in range(len(e)):
            e[j][i] = e[j][i]/s

    #保存q与e
    with open('data/qmatrix', 'w', encoding='UTF-8') as f4:
        for i in range(len(q)):
            for j in range(len(q[i])):
                f4.write(str(q[i][j]))
                f4.write(' ')
            f4.write('\n')
    with open('data/ematrix', 'w', encoding='UTF-8') as f5:
        for i in range(len(e)):
            for j in range(len(e[i])):
                f5.write(str(e[i][j]))
                f5.write(' ')
            f5.write('\n')
