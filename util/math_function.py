import torch


# step function
def step_f(x):
    return 0 if x < 0 else 1


# 双曲log用于将实际偏移转换成最后的网络输出
def hyperbola_log(x):
    return step_f(-x) * torch.log(x + 1) + step_f(x) * (-torch.log(-x + 1))


def hyperbola_sqrt(x):
    if x < 0:
        return -torch.sqrt(-x)
    else:
        return torch.sqrt(x)


def hyperbola_cube_root(x, alpha):
    """
    if x < 0:
        return -torch.pow(-x, alpha)
    else:
        return torch.pow(x, alpha)"""
    x = torch.where(x < 0, -torch.pow(-x, alpha), x)
    x = torch.where(x >= 0, torch.pow(x, alpha), x)
    return x


if __name__ == '__main__':
    x = torch.randn((3, 4, 2))
    print(x)
    y = hyperbola_cube_root(x, 1/3)
    print(y)

