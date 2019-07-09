# --encoding:utf-8 --
"""训练分词用的HMM"""

import numpy as np
import math
import random

from hmm2 import hmm_learn

random.seed(28)
infinite = float(-2 ** 31)


def log_normalize(a):
    s = 0
    for i in a:
        s += i
    s = math.log(s)
    for i in range(len(a)):
        if a[i] == 0:
            a[i] = infinite
        else:
            a[i] = math.log(a[i]) - s


def fit(train_file_path, mode='r', encoding='utf-8'):
    """
    进行模型训练，并返回pi、A、B
    :param train_file_path:
    :return:
    """
    # 1. 加载数据
    with open(train_file_path, mode=mode, encoding=encoding) as reader:
        # 读取所有数据（因为数据格式第一个字符是不可见字符<文件描述符>）
        sentence = reader.read()[1:]

    # 1. 初始化pi、A、B
    pi = np.zeros(4)
    A = np.zeros((4, 4))
    B = np.zeros((4, 65536))

    # 2. 模型训练（使用MLE来预测） 0B/1M/2E/3S
    tokens = sentence.split(' ')
    last_i = 2  # 上一个词结束的状态
    for k, token in enumerate(tokens):
        token = token.strip()
        n = len(token)
        if n <= 0:
            continue

        if n == 1:
            pi[3] += 1
            A[last_i][3] += 1
            B[3][ord(token[0])] += 1
            last_i = 3
            continue

        # 初始化向量
        pi[0] += 1  # 作为开始
        pi[2] += 1  # 作为结束
        pi[1] += (n - 2)  # 中间词数目

        # 转移矩阵
        A[last_i][0] += 1
        last_i = 2
        if n == 2:
            A[0][2] += 1
        else:
            A[0][1] += 1
            A[1][1] += (n - 3)
            A[1][2] += 1

        # 发射矩阵
        B[0][ord(token[0])] += 1
        B[2][ord(token[n - 1])] += 1
        for i in range(1, n - 1):
            B[1][ord(token[i])] += 1

    # 正则化
    log_normalize(pi)
    for i in range(4):
        log_normalize(A[i])
        log_normalize(B[i])

    # 结果返回
    return pi, A, B


def dump(pi, A, B):
    """
    模型保存
    :param pi:
    :param A:
    :param B:
    :return:
    """
    n, m = np.shape(B)

    # 1. pi输出
    with open("pi.txt", "w") as f_pi:
        f_pi.write(str(n))
        f_pi.write('\n')
        f_pi.write(' '.join(map(str, pi)))

    # 2. A输出
    with open('A.txt', 'w') as f_a:
        f_a.write(str(n))
        f_a.write('\n')
        for a in A:
            f_a.write(' '.join(map(str, a)))
            f_a.write('\n')

    # 3. B输出
    with open('B.txt', 'w') as f_b:
        f_b.write(str(n))
        f_b.write('\n')
        f_b.write(str(m))
        f_b.write('\n')
        for b in B:
            f_b.write(' '.join(map(str, b)))
            f_b.write('\n')


def load():
    """
    模型加载
    :return:
    """
    with open('pi.txt', 'r', encoding='utf-8') as f_pi:
        f_pi.readline()  # 第一行不需要
        line = f_pi.readline()
        pi = list(map(float, line.strip().split(' ')))

    with open('A.txt', 'r', encoding='utf-8') as f_a:
        n = int(f_a.readline())
        A = np.zeros((n, n))
        i = 0
        for line in f_a:
            j = 0
            for v in map(float, line.strip().split(' ')):
                A[i][j] = v
                j += 1
            i += 1

    with open('B.txt', 'r', encoding='utf-8') as f_b:
        n = int(f_b.readline())
        m = int(f_b.readline())
        B = np.zeros((n, m))
        i = 0
        for line in f_b:
            j = 0
            for v in map(float, line.strip().split(' ')):
                B[i][j] = v
                j += 1
            i += 1

    return pi, A, B


def segment(sentence, decode):
    """
    分词
    :param sentence:
    :param decode:
    :return:
    """
    T = len(sentence)
    i = 0
    while i < T:  # B/M/E/S
        if decode[i] == 0 or decode[i] == 1:  # Begin
            j = i + 1
            while j < T:
                if decode[j] == 2:
                    break
                j += 1
            print(sentence[i:j + 1], end=' | ')
        elif decode[i] == 3 or decode[i] == 2:  # single
            print(sentence[i:i + 1], end=' | ')
        else:
            print("Error")
        i += 1


if __name__ == '__main__':
    # 1. 模型训练
    # 该数据默认是四个状态"B/M/E/S"
    # B:begin, M:middle, E:end, S:single
    # 分别代表每个状态代表的是该字在词语中的位置，B代表该字是词语中的起始字，M代表是词语中的中间字，E代表是词语中的结束字，S则代表是单字成词。
    # pi, A, B = fit('pku_training.utf8')
    # print(pi)
    # print(A)
    # print(B)

    # # 2. 模型输出
    # dump(pi, A, B)

    # 3. 模型加载
    pi, A, B = load()
    print(pi)
    print(A)
    print(B)

    # 4. 进行分词操作
    with open('novel.txt', 'r', encoding='utf-8') as reader:
        data = reader.read()[1:]
    decode = hmm_learn.viterbi(pi, A, B, data)
    segment(data, decode)
