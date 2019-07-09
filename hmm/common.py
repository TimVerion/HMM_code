# --encoding:utf-8 --
"""公用的函数"""


def convert_obs_seq_2_index(Q, index=None):
    """
    将观测序列转换为观测值的索引值
    Q:是输入的观测序列
    """
    if index is not None:
        cht = Q[index]
        if "黑" == cht:
            return 1
        else:
            return 0
    else:
        result = []
        for q in Q:
            if "黑" == q:
                result.append(1)
            else:
                result.append(0)
        return result
