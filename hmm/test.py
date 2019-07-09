from hmm import common
import numpy as np


def col_forward(pi, A, B, Q, alpha, fetch_bij=common.convert_obs_seq_2_index):
    # 更新初始值
    for i in range(alpha.shape[1]):
        alpha[0][i] = pi[i] * B[i][fetch_bij(Q, 0)]
    # 递推
    for t in range(1, alpha.shape[0]):
        for i in range(len(B)):
            tem = 0
            for j in range(len(A)):
                tem += alpha[t - 1][j] * A[j][i]
            alpha[t][i] = tem * B[i][fetch_bij(Q, t)]
    return alpha


def col_back(pi, A, B, Q, beta, fetch_bji=common.convert_obs_seq_2_index):
    T = len(Q)
    for i in range(beta.shape[1]):
        beta[T - 1][i] = 1
    for t in range(T - 2, -1, -1):
        for i in range(len(A)):
            tem = 0
            for j in range(len(B)):
                tem += A[i][j] * B[j][fetch_bji(Q, t + 1)]*beta[t+1,j]
            beta[t][i] = tem
    return beta


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
    Q = '白黑白白黑'
    alpha = np.zeros((len(Q), len(A)))
    # 开始计算
    alpha = col_forward(pi, A, B, Q, alpha)
    # 输出最终结果
    print("***********前向算法*************")
    print(alpha)
    # 计算最终概率值：
    p = 0
    for i in alpha[-1]:
        p += i
    print(Q, end="->出现的概率为:")
    print(p)
    print("***********反向算法*************")
    beta = np.zeros((len(Q), len(A)))
    beta = col_back(pi, A, B, Q, beta)
    p = 0
    print(beta)
    for i in range(len(A)):
        p += pi[i]*B[i][common.convert_obs_seq_2_index(Q,0)]*beta[0][i]
    print(Q, end="->出现的概率为:")
    print(p)
