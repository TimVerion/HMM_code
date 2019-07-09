# --encoding:utf-8 --
"""课堂上实现的一个计算HMM模型中前向概率的一个算法实现(前向算法)"""

import numpy as np


def calc_alpha(pi, A, B, Q, alpha):
    """
    计算前向概率alpha，并将结果保存到alpha矩阵中(T*n)
    :param pi:  1*n
    :param A:  n*n
    :param B:  n*m
    :param Q:  1*T => T的数组
    :param alpha: T*n
    :return: alpha
    """
    # 1. 获取相关的变量信息
    n = np.shape(A)[0]
    T = np.shape(Q)[0]

    # 2 更新t=1时刻的前向概率值
    for i in range(n):
        alpha[0][i] = pi[i] * B[i][Q[0]]

    # 3. 更新t=2... T时刻对应的前向概率值
    tmp = np.zeros(n)
    for t in range(1, T):
        # 迭代计算t时刻对应的前向概率值
        for i in range(n):
            # 计算时刻t状态为i的前向概率
            # a. 计算上一个时刻t-1累计到当前状态i的概率值
            for j in range(n):
                tmp[j] = alpha[t - 1][j] * A[j][i]

            # b. 更新时刻t对应状态i的前向概率值
            alpha[t][i] = np.sum(tmp) * B[i][Q[t]]

    # 4. 返回结果
    return alpha


if __name__ == '__main__':
    # 测试
    pi = np.array([0.2, 0.5, 0.3])
    A = np.array([
        [0.5, 0.4, 0.1],
        [0.2, 0.2, 0.6],
        [0.2, 0.5, 0.3]
    ])
    B = np.array([
        [0.4, 0.6],
        [0.8, 0.2],
        [0.5, 0.5]
    ])
    # 白，黑，白，白，黑
    Q = [0, 1, 0, 0, 1]
    alpha = np.zeros((len(Q), len(A)))
    # 开始计算
    calc_alpha(pi, A, B, Q, alpha)
    # 输出最终结果
    print(alpha)

    # 计算最终概率值：
    p = 0
    for i in alpha[-1]:
        p += i
    print(Q, end="->出现的概率为:")
    print(p)
