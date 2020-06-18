import numpy as np
#viterbi算法
def viterbi(sentindex, tagset, q, e):
    u = 0
    cur = 1
    predicttag = []
    for i in range(len(sentindex)):
        lis = [cur*q[u][tag]*e[sentindex[i]][tag-1] for tag in tagset]
        cur = max(lis)
        lis = np.array(lis)
        u = np.argmax(lis)+1
        predicttag.append(u)
    return predicttag

#评估预测结果
def eva(predict, target):
    s = sum([len(elem) for elem in predict])
    right = 0
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            if predict[i][j]==target[i][j]:
                right+=1
    return right/s
if __name__ == "__main__":
    #分别读取word2index,tag2index,q,e
    with open('data/word2index', 'r', encoding='UTF-8') as f1:
        pairlis = f1.readlines()
    word2index = dict()
    for pair in pairlis:
        temp = pair.strip().split(' ')
        word2index[temp[0]] = int(temp[1])
    with open('data/tag2index', 'r', encoding='UTF-8') as f2:
        pairlis = f2.readlines()
    tag2index = dict()
    for pair in pairlis:
        temp = pair.strip().split(' ')
        tag2index[temp[0]] = int(temp[1])
    with open('data/qmatrix', 'r', encoding='UTF-8') as f3:
        lines = f3.readlines()
    q = []
    for line in lines:
        q.append([float(i) for i in line.strip().split(' ')])
    with open('data/ematrix', 'r', encoding='UTF-8') as f4:
        lines = f4.readlines()
    e = []
    for line in lines:
        e.append([float(i) for i in line.strip().split(' ')])

    #读取待测试集,并数字化
    with open('data/dev.conll', 'r', encoding='UTF-8') as f5:
        lis = [line.strip().split('\t') for line in f5.readlines()]
    text = []
    sent = []
    for line in lis:
        if line == ['']:
            text.append(sent)
            sent = []
        else:
            sent.append((line[1], line[3]))
    text = [[(word2index.get(pair[0], -1), tag2index[pair[1]]) for pair in sent] for sent in text]

    #生成句子索引序列，标注索引序列,标注集
    sentindexset = [[pair[0] for pair in sent]for sent in text]
    targetset = [[pair[1] for pair in sent] for sent in text]
    tagset = list(tag2index.values())
    #对每句话进行预测，生成预测集合
    predict = []
    for sentindex in sentindexset:
        predict.append(viterbi(sentindex, tagset, q, e))
    #计算准确率
    x = eva(predict, targetset)
    print('准确率:', x)
