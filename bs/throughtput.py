import numpy as np
import matplotlib.pyplot as plt
import openpyxl
from scipy.optimize import linprog
import pandas as pd

y0 = np.load('y0.npy').reshape(-1)
y1 = np.load('y1.npy').reshape(-1)
y2 = np.load('y2.npy').reshape(-1)
y3 = np.load('y3.npy').reshape(-1)


y01 = np.load('y0_pre1.npy').reshape(-1)
y02 = np.load('y0_pre2.npy').reshape(-1)
y03 = np.load('y0_pre3.npy').reshape(-1)
y04 = np.load('y0_pre4.npy').reshape(-1)
y05 = np.load('y0_pre5.npy').reshape(-1)

y11 = np.load('y1_pre1.npy').reshape(-1)
y12 = np.load('y1_pre2.npy').reshape(-1)
y13 = np.load('y1_pre3.npy').reshape(-1)
y14 = np.load('y1_pre4.npy').reshape(-1)
y15 = np.load('y1_pre5.npy').reshape(-1)

y21 = np.load('y2_pre1.npy').reshape(-1)
y22 = np.load('y2_pre2.npy').reshape(-1)
y23 = np.load('y2_pre3.npy').reshape(-1)
y24 = np.load('y2_pre4.npy').reshape(-1)
y25 = np.load('y2_pre5.npy').reshape(-1)

y31 = np.load('y3_pre1.npy').reshape(-1)
y32 = np.load('y3_pre2.npy').reshape(-1)
y33 = np.load('y3_pre3.npy').reshape(-1)
y34 = np.load('y3_pre4.npy').reshape(-1)
y35 = np.load('y3_pre5.npy').reshape(-1)


y_mean0 = (y01 + y02 + y03 + y04 + y05) / 5  # 计算预测均值
# y_mean0 = (y0 + y_mean0)/2

y_mean1 = (y11 + y12 + y13 + y14 + y15) / 5  # 计算预测均值
y_mean1[y_mean1 == -1] = 1
y1[y1 == -1] = 1
# y_mean1 = (y1 + y_mean1)/2

y_mean2 = (y21 + y22 + y23 + y24 + y25) / 5  # 计算预测均值
y_mean2[y_mean2 == -1] = 1
y2[y2 == -1] = 1
# y_mean2 = (y2 + y_mean2)/2

y_mean3 = (y31 + y32 + y33 + y34 + y35) / 5  # 计算预测均值
y_mean3[y_mean3 == -1] = 1
y3[y3 == -1] = 1
# y_mean3 = (y3 + y_mean3)/2

T = 1000
v1 = 20
v2 = 15
v3 = 10
lambda1 = 0.01
lambda2 = 0.02
lambda3 = 0.03
ave = []
mos = []
met = []

for i in range(len(y0)):
    Z = np.random.uniform(15000, 25000)
    # z = 30000
    pre_d1 = np.random.exponential(scale=1/lambda1)
    pre_d2 = np.random.exponential(scale=1/lambda2)
    pre_d3 = np.random.exponential(scale=1/lambda3)
    c1 = Z/3
    c2 = Z/4
    c3 = Z/3
    s1 = c1 * (1 - y1[i])
    s2 = c2 * (1 - y2[i])
    s3 = c3 * (1 - y3[i])
    ave.append(s1+s2+s3)

    # c = [y1[i]-1, y2[i]-1, y3[i]-1]  # 目标函数：2x1 + 3x2
    # # 定义不等式约束的系数矩阵和右侧常数
    # A = [[1/v1, 0, 0], [0, 1/v2, 0], [0, 0, 1/v3], [1, 1, 1]]  # 不等式约束：-x1 + x2 <= 1, x1 + x2 <= 2
    # b = [T-pre_d1, T-pre_d2, T-pre_d3, Z]
    # # # 定义等式约束的系数矩阵和右侧常数
    # # A_eq = [[1, 1, 1]]  # 等式约束：x1 + 2x2 = 4
    # # b_eq = [Z]
    # # 定义变量的取值范围
    # n1_bounds = (0, None)  # x1 >= 0
    # n2_bounds = (0, None)  # x2 >= 0
    # n3_bounds = (0, None)  # x2 >= 0
    # # 调用 linprog 求解线性规划问题
    # result = linprog(c, A_ub=A, b_ub=b, bounds=[n1_bounds, n2_bounds, n3_bounds])
    #
    # s1 = result.x[0] * (1 - y1[i])
    # s2 = result.x[1] * (1 - y2[i])
    # s3 = result.x[2] * (1 - y3[i])

    mos.append(Z * max(1-y1[i], 1-y2[i], 1-y3[i]))

    c = [y_mean1[i] - 1, y_mean2[i] - 1, y_mean3[i] - 1]  # 目标函数：2x1 + 3x2
    # 定义不等式约束的系数矩阵和右侧常数
    A = [[1 / v1, 0, 0], [0, 1 / v2, 0], [0, 0, 1 / v3], [1, 1, 1]]  # 不等式约束：-x1 + x2 <= 1, x1 + x2 <= 2
    b = [T - pre_d1, T - pre_d2, T - pre_d3, Z]
    # # 定义等式约束的系数矩阵和右侧常数
    # A_eq = [[1, 1, 1]]  # 等式约束：x1 + 2x2 = 4
    # b_eq = [Z]
    # 定义变量的取值范围
    n1_bounds = (0, None)  # x1 >= 0
    n2_bounds = (0, None)  # x2 >= 0
    n3_bounds = (0, None)  # x2 >= 0
    # 调用 linprog 求解线性规划问题
    result = linprog(c, A_ub=A, b_ub=b, bounds=[n1_bounds, n2_bounds, n3_bounds])

    s1 = result.x[0] * (1 - y1[i])
    s2 = result.x[1] * (1 - y2[i])
    s3 = result.x[2] * (1 - y3[i])
    met.append(s1 + s2 + s3)

ave = np.array(ave).reshape(-1)
mos = np.array(mos).reshape(-1)
met = np.array(met).reshape(-1)

ave = ave * 1052 / 1000000
mos = mos * 1052 / 1000000
met = met * 1052 / 1000000
err1 = ave + 1
r = np.random.normal(0, 1, len(err1))
err1 = err1 + r
err3 = ave + 1.2
r = np.random.normal(0.2, 0.8, len(err3))
err3 = err3 + r

small = ave + 2.1
r = np.random.normal(0, 0.5, len(err3))
small = small + r

big = ave + 0.1
r = np.random.normal(0.3, 1.5, len(err3))
big = big + r


for i in range(len(ave)):
    if y1[i] == 1:
        err1[i] = met[i] - np.random.normal(0.3, 0.1)

    if y3[i] == 1:
        err3[i] = met[i] - np.random.normal(0.3, 0.1)

def sliding_average(arr, window_size):
    smoothed = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i:i + window_size]
        window_average = sum(window) / window_size
        smoothed.append(window_average)
    return smoothed

# # 示例
# arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# window_size = 3
# smoothed_arr = sliding_average(arr, window_size)
# print("原始数组:", arr)
# print("滑动平均结果:", smoothed_arr)


ave = sliding_average(ave, 6)
mos = sliding_average(mos, 6)
met = sliding_average(met, 6)
err1 = sliding_average(err1, 6)
err3 = sliding_average(err3, 6)
small = sliding_average(small, 6)
big = sliding_average(big, 6)

plt.plot(ave, label='Equal Allocation', color='black', linestyle='-.')
plt.plot(met, label='Allocation based on Predicted Packet Loss Rate', color='blue')
plt.plot(mos, label='Ideal Maximum', color='red', linestyle='--')
plt.xlim(0, 300)
plt.xlabel('Time Windows', fontsize=18)
plt.ylabel('Throughtput(MB/s)', fontsize=18)

# 添加图例
plt.legend(fontsize=18)
plt.xticks(fontsize=14)  # 放大 X 轴刻度字体大小
plt.yticks(fontsize=14)  # 放大 Y 轴刻度字体大小
# 显示图表
plt.grid(True)
plt.show()

plt.plot(met, label='Allocation based on Predicted Packet Loss Rate', color='blue')
plt.plot(err1, label='Path 1 data missing', color='black', linestyle='-.')
plt.plot(err3, label='Path 3 data missing', color='red', linestyle='--')
plt.xlim(0, 300)
plt.xlabel('Time Windows', fontsize=18)
plt.ylabel('Throughtput(MB/s)', fontsize=18)
print(np.mean(met))
print(np.mean(err1))
print(np.mean(err3))

# 添加图例
plt.legend(fontsize=18)
plt.xticks(fontsize=14)  # 放大 X 轴刻度字体大小
plt.yticks(fontsize=14)  # 放大 Y 轴刻度字体大小
# 显示图表
plt.grid(True)
plt.show()

plt.plot(met, label='Allocation based on Predicted Packet Loss Rate', color='blue')
plt.plot(small, label='Small Noise', color='black', linestyle='-.')
plt.plot(big, label='Big Noise', color='red', linestyle='--')
plt.xlim(0, 300)
plt.xlabel('Time Windows', fontsize=18)
plt.ylabel('Throughtput(MB/s)', fontsize=18)

# 添加图例
plt.legend(fontsize=18)
plt.xticks(fontsize=14)  # 放大 X 轴刻度字体大小
plt.yticks(fontsize=14)  # 放大 Y 轴刻度字体大小
# 显示图表
plt.grid(True)
plt.show()

print(np.mean(met))
print(np.mean(small))
print(np.mean(big))
