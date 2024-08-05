import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from scipy.optimize import linprog
import pandas as pd
# 不同的指数分布参数
# lambda_values = [0.05, 0.03, 0.02]
#
# # 生成横坐标值（随机变量的取值范围）
# x = np.linspace(0, 500)
#
# # 绘制不同参数指数分布的概率密度图
# for lam in lambda_values:
#     y = lam * np.exp(-lam * x)
#     plt.plot(x, y, label=f'lambda={lam}')
#
# # 设置图例、标签和标题
# plt.legend()
# plt.xlabel('随机变量的取值')
# plt.ylabel('概率密度')
# plt.title('不同参数指数分布的概率密度图')
#
# # 显示图形
# plt.show()

T = 1000

v1 = 20
v2 = 15
v3 = 10

lambda1 = 0.01
lambda2 = 0.02
lambda3 = 0.03

a1 = 0.05
b1 = 0.45
a2 = 0.15
b2 = 0.35
a3 = 0.20
b3 = 0.30

mu1 = 17
sig1 = 9
mu2 = 17.5
sig2 = 6
mu3 = 18
sig3 = 3

m = 1000

data1 = []
data2 = []
data3 = []

file_path = 'D:\M4_1\\bs\\20240122_BER.xlsx'  # 请替换为您的文件路径
workbook = openpyxl.load_workbook(file_path)
sheet = workbook.active

# 获取SNR和BER数据
snr_row = [cell.value for cell in sheet[1]]
ber_row = [cell.value for cell in sheet[2]]

# 将数据存储在NumPy数组中
snr_array = np.array(snr_row[1:])
ber_array = np.array(ber_row[1:])
plr_array = 1 - (ber_array - 1) ** (1518*8)

degree = 15  # 选择多项式的次数
coefficients = np.polyfit(snr_array, plr_array, degree)
poly = np.poly1d(coefficients)

# # 生成拟合曲线的数据
# snr_fit = np.linspace(min(snr_array), max(snr_array), 100)
# plr_fit = poly(snr_fit)
#
# # 绘制原始数据和拟合曲线
# plt.scatter(snr_array, plr_array, label='Original Data')
# plt.plot(snr_fit, plr_fit, label=f'Polynomial Fit (Degree {degree})', color='r')
#
# # 添加标题和标签
# plt.title('Polynomial Fit of PLR vs SNR')
# plt.xlabel('SNR')
# plt.ylabel('PLR')
#
# # 添加网格和图例
# plt.grid(True)
# plt.legend()
#
# # 显示图形
# plt.show()
#
# # 输出snr=10.5时plr的值
# snr_value = 10.5
# plr_value = poly(snr_value)
# print(f"PLR at SNR {snr_value}: {plr_value}")

# # 绘制折线图
# plt.plot(snr_array, plr_array, marker='o', linestyle='-', color='b', label='PLR vs SNR')
# # 添加标题和标签
# plt.title('Packet Loss Rate vs Signal-to-Noise Ratio')
# plt.xlabel('SNR')
# plt.ylabel('PLR')
# # 添加网格
# plt.grid(True)
# # 添加图例
# plt.legend()
# # 显示图形
# plt.show()


def wheloss(snr, poly):
    p = poly(snr)
    # p = plr_array[int(snr)]
    x = np.random.rand()
    if x > p:
        return 1
    else:
        return 0


for l in range(1, m+1):
# 生成服从均匀分布的随机值
    pre_p1 = np.random.uniform(a1, b1)
    pre_p2 = np.random.uniform(a2, b2)
    pre_p3 = np.random.uniform(a3, b3)
    Z = np.random.uniform(18000, 38000)
    # 生成服从指数分布的随机值
    pre_d1 = np.random.exponential(scale=1/lambda1)
    pre_d2 = np.random.exponential(scale=1/lambda2)
    pre_d3 = np.random.exponential(scale=1/lambda3)


    # 定义线性规划问题的目标函数系数
    c = [pre_p1-1, pre_p2-1, pre_p3-1]  # 目标函数：2x1 + 3x2

    # 定义不等式约束的系数矩阵和右侧常数
    A = [[1/v1, 0, 0], [0, 1/v2, 0], [0, 0, 1/v3], [1, 1, 1]]  # 不等式约束：-x1 + x2 <= 1, x1 + x2 <= 2
    b = [T-pre_d1, T-pre_d2, T-pre_p3, Z]

    # # 定义等式约束的系数矩阵和右侧常数
    # A_eq = [[1, 1, 1]]  # 等式约束：x1 + 2x2 = 4
    # b_eq = [Z]

    # 定义变量的取值范围
    n1_bounds = (0, None)  # x1 >= 0
    n2_bounds = (0, None)  # x2 >= 0
    n3_bounds = (0, None)  # x2 >= 0

    # 调用 linprog 求解线性规划问题
    result = linprog(c, A_ub=A, b_ub=b, bounds=[n1_bounds, n2_bounds, n3_bounds])

    # 输出结果
    # print("最小值:", result.fun)
    # print("最优解:", result.x)
    # print(int(result.x[0]))
    # print(int(result.x[1]))
    # print(int(result.x[2]))
    receive = 0
    avesnr = 0
    avedelay = 0
    if int(result.x[0]) == 0:
        data1.append([-1, -1, -1])
        receive1 = 0
    else:
        for j in range(1, int(result.x[0])+1):
            snr = np.random.normal(mu1, sig1)
            avesnr = avesnr + snr
            delay = np.random.exponential(scale=1/lambda1)
            avedelay = avedelay + delay
            if j/v1 + 2 * delay <= T and wheloss(snr, poly) == 1:
                receive = receive + 1
        loss = 1 - receive/int(result.x[0])
        receive1 = receive
        avedelay = avedelay/int(result.x[0])
        avesnr = avesnr/int(result.x[0])
        data1.append([avedelay, avesnr, loss])

    receive = 0
    avesnr = 0
    avedelay = 0
    if int(result.x[1]) == 0:
        data2.append([-1, -1, -1])
        receive2 = 0
    else:
        for j in range(1, int(result.x[1])+1):
            snr = np.random.normal(mu2, sig2)
            avesnr = avesnr + snr
            delay = np.random.exponential(scale=1/lambda2)
            avedelay = avedelay + delay
            if j/v2 + 2 * delay <= T and wheloss(snr, poly) == 1:
                receive = receive + 1
        loss = 1 - receive/int(result.x[1])
        receive2 = receive
        avedelay = avedelay/int(result.x[1])
        avesnr = avesnr/int(result.x[1])
        data2.append([avedelay, avesnr, loss])

    receive = 0
    avesnr = 0
    avedelay = 0
    if int(result.x[2]) == 0:
        multiloss = 1 - (receive1 + receive2)/(result.x[0] + result.x[1])
        data3.append([-1, -1, -1, multiloss])
    else:
        for j in range(1, int(result.x[2])+1):
            snr = np.random.normal(mu3, sig3)
            avesnr = avesnr + snr
            delay = np.random.exponential(scale=1/lambda3)
            avedelay = avedelay + delay
            if j/v3 + 2 * delay <= T and wheloss(snr, poly) == 1:
                receive = receive + 1
        loss = 1 - receive/int(result.x[2])
        receive3 = receive
        avedelay = avedelay/int(result.x[2])
        avesnr = avesnr/int(result.x[2])
        multiloss = 1 - (receive1 + receive2 + receive3)/(result.x[0] + result.x[1] + result.x[2])
        data3.append([avedelay, avesnr, loss, multiloss])

# 将列表转换为DataFrame
df = pd.DataFrame(data1)
# 保存DataFrame为Excel文件
df.to_excel('data1.xlsx', index=False)
# 将列表转换为DataFrame
df = pd.DataFrame(data2)
# 保存DataFrame为Excel文件
df.to_excel('data2.xlsx', index=False)
# 将列表转换为DataFrame
df = pd.DataFrame(data3)
# 保存DataFrame为Excel文件
df.to_excel('data3.xlsx', index=False)









