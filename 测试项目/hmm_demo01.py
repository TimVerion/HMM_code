# --encoding:utf-8 --
"""HMM算法模型应用案例01"""

import numpy as np
from hmm2 import hmm_learn, common

if __name__ == '__main__':
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
    # 对A\B\pi进行log转换、
    pi = np.log(pi)
    A = np.log(A)
    B = np.log(B)

    Q = ['白', '黑', '白', '白', '黑']
    T = len(Q)
    n = len(A)

    print("测试前向概率计算....................")
    alpha = np.zeros((T, n))
    # 开始计算
    hmm_learn.calc_alpha(pi, A, B, Q, alpha, common.convert_obs_seq_2_index)
    # 输出最终结果
    # print(np.exp(alpha))
    print(alpha)

    # 计算最终概率值：
    p = common.log_sum_exp(alpha[T - 1].flat)
    print(Q, end="->出现的概率为:")
    print(np.exp(p))

    print("测试后向概率计算....................")
    beta = np.zeros((T, n))
    # 开始计算
    hmm_learn.calc_beta(pi, A, B, Q, beta, common.convert_obs_seq_2_index)
    # 输出最终结果
    # print(np.exp(beta))
    print(beta)

    # 计算最终概率值：
    tmp_p = np.zeros(n)
    for i in range(n):
        tmp_p[i] = pi[i] + B[i][common.convert_obs_seq_2_index(Q, 0)] + beta[0][i]
    print("--------------")
    p = common.log_sum_exp(tmp_p)
    print(Q, end="->出现的概率为:")
    print(np.exp(p))

    print("测试gamma矩阵应用在最大概率预测的情况下.....................")
    gamma = np.zeros((T, n))
    hmm_learn.calc_gamma(alpha, beta, gamma)
    # print(np.exp(gamma))
    print(gamma)
    print("各个时刻最大概率的盒子为:", end='')
    index = ['盒子1', '盒子2', '盒子3']
    for p in gamma:
        print(index[p.tolist().index(np.max(p))], end="\t")
    print()

    print("测试ksi矩阵....................")
    ksi = np.zeros((T - 1, n, n))
    hmm_learn.calc_ksi(alpha, beta, A, B, Q, ksi, common.convert_obs_seq_2_index)
    print(ksi)

    print("baum welch算法应用.......................")
    hmm_learn.baum_welch(pi, A, B, Q, max_iter=3, fetch_index_by_obs_seq=common.convert_obs_seq_2_index)
    print("最终的pi矩阵：", end='')
    print(np.exp(pi))
    print("最终的状态转移矩阵：")
    print(np.exp(A))
    print("最终的状态-观测值转移矩阵：")
    print(np.exp(B))

    print("viterbi算法应用.....................")
    state_seq = hmm_learn.viterbi(pi, A, B, Q, common.convert_obs_seq_2_index)
    print("最终结果为:", end='')
    print(state_seq)
    state = ['盒子1', '盒子2', '盒子3']
    for i in state_seq:
        print(state[i], end='\t')

    print("\n根据当前的状态值，预测未来的状态值....................")
    ### 假定当前状态为: 盒子1，那么接下来4次的状态分别是多少
    ### 假定当前状态为: 盒子1, 盒子2，那么接下来3次的状态分别是多少
    ### 假定当前状态为: 盒子1, 盒子3，那么接下来3次的状态分别是多少
    ### 假定当前状态为: 盒子1, 盒子1，那么接下来3次的状态分别是多少
    current_state = [0, 0]
    for t in range(1, 4):
        current_i = current_state[-1]
        max_ksi_j = ksi[t][current_i][0]
        max_j = 0
        for j in range(1, n):
            tmp = ksi[t][current_i][j]
            if tmp > max_ksi_j:
                max_ksi_j = tmp
                max_j = j

        current_state.append(max_j)
    print(current_state)
