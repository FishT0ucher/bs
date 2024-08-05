import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

e0 = np.load('e0.npy').reshape(-1)
e1 = np.load('e1.npy').reshape(-1)
e2 = np.load('e2.npy').reshape(-1)
e3 = np.load('e3.npy').reshape(-1)

ee0 = np.load('ee0.npy').reshape(-1)
ee1 = np.load('ee1.npy').reshape(-1)
ee2 = np.load('ee2.npy').reshape(-1)
ee3 = np.load('ee3.npy').reshape(-1)

e03 = np.load('e03.npy').reshape(-1)
e13 = np.load('e13.npy').reshape(-1)
e23 = np.load('e23.npy').reshape(-1)
e33 = np.load('e33.npy').reshape(-1)

ee03 = np.load('ee03.npy').reshape(-1)
ee13 = np.load('ee13.npy').reshape(-1)
ee23 = np.load('ee23.npy').reshape(-1)
ee33 = np.load('ee33.npy').reshape(-1)




# plt.plot(e0, label='Original Model', color='blue')
# plt.plot(ee0, label='Model With Structural Change', color='red', linestyle='--')
#
# plt.xlim(0, 300)  # x 轴显示范围从 0 到 10
# # plt.ylim(-0.5, 0.5)  # y 轴显示范围从 -1 到 1
#
# # 添加标题和标签
# plt.xlabel('Time Windows', fontsize=18)
# plt.ylabel('Prediction Error', fontsize=18)
#
# # 添加图例
# plt.legend(fontsize=18)
# plt.xticks(fontsize=14)  # 放大 X 轴刻度字体大小
# plt.yticks(fontsize=14)  # 放大 Y 轴刻度字体大小
# # 显示图表
# plt.grid(True)
# plt.show()
#
#
# plt.plot(e1, label='Original Model', color='blue')
# plt.plot(ee1, label='Model With Structural Change', color='red', linestyle='--')
#
# plt.xlim(0, 300)  # x 轴显示范围从 0 到 10
# # plt.ylim(-0.5, 0.5)  # y 轴显示范围从 -1 到 1
#
# # 添加标题和标签
# plt.xlabel('Time Windows', fontsize=18)
# plt.ylabel('Prediction Error', fontsize=18)
#
# # 添加图例
# plt.legend(fontsize=18)
# plt.xticks(fontsize=14)  # 放大 X 轴刻度字体大小
# plt.yticks(fontsize=14)  # 放大 Y 轴刻度字体大小
# # 显示图表
# plt.grid(True)
# plt.show()
#
plt.plot(e2, label='Original Model', color='blue')
plt.plot(ee2, label='Model With Structural Change', color='red', linestyle='--')

plt.xlim(0, 300)  # x 轴显示范围从 0 到 10
# plt.ylim(-0.5, 0.5)  # y 轴显示范围从 -1 到 1

# 添加标题和标签
plt.xlabel('Time Windows', fontsize=18)
plt.ylabel('Prediction Error', fontsize=18)

# 添加图例
plt.legend(fontsize=18)
plt.xticks(fontsize=14)  # 放大 X 轴刻度字体大小
plt.yticks(fontsize=14)  # 放大 Y 轴刻度字体大小
# 显示图表
plt.grid(True)
plt.show()
#
# plt.plot(e3, label='Original Model', color='blue')
# plt.plot(ee3, label='Model With Structural Change', color='red', linestyle='--')
#
# plt.xlim(0, 300)  # x 轴显示范围从 0 到 10
# # plt.ylim(-0.5, 0.5)  # y 轴显示范围从 -1 到 1
#
# # 添加标题和标签
# plt.xlabel('Time Windows', fontsize=18)
# plt.ylabel('Prediction Error', fontsize=18)
#
# # 添加图例
# plt.legend(fontsize=18)
# plt.xticks(fontsize=14)  # 放大 X 轴刻度字体大小
# plt.yticks(fontsize=14)  # 放大 Y 轴刻度字体大小
# # 显示图表
# plt.grid(True)
# plt.show()




#
# plt.plot(e03, label='Original Model', color='blue')
# plt.plot(ee03, label='Model With Structural Change', color='red', linestyle='--')
#
# plt.xlim(0, 300)  # x 轴显示范围从 0 到 10
# # plt.ylim(-0.5, 0.5)  # y 轴显示范围从 -1 到 1
#
# # 添加标题和标签
# plt.xlabel('Time Windows', fontsize=18)
# plt.ylabel('Prediction Error', fontsize=18)
#
# # 添加图例
# plt.legend(fontsize=18)
# plt.xticks(fontsize=14)  # 放大 X 轴刻度字体大小
# plt.yticks(fontsize=14)  # 放大 Y 轴刻度字体大小
# # 显示图表
# plt.grid(True)
# plt.show()
#
#
# plt.plot(e13, label='Original Model', color='blue')
# plt.plot(ee13, label='Model With Structural Change', color='red', linestyle='--')
#
# plt.xlim(0, 300)  # x 轴显示范围从 0 到 10
# # plt.ylim(-0.5, 0.5)  # y 轴显示范围从 -1 到 1
#
# # 添加标题和标签
# plt.xlabel('Time Windows', fontsize=18)
# plt.ylabel('Prediction Error', fontsize=18)
#
# # 添加图例
# plt.legend(fontsize=18)
# plt.xticks(fontsize=14)  # 放大 X 轴刻度字体大小
# plt.yticks(fontsize=14)  # 放大 Y 轴刻度字体大小
# # 显示图表
# plt.grid(True)
# plt.show()
#
# plt.plot(e23, label='Original Model', color='blue')
# plt.plot(ee23, label='Model With Structural Change', color='red', linestyle='--')
#
# plt.xlim(0, 300)  # x 轴显示范围从 0 到 10
# # plt.ylim(-0.5, 0.5)  # y 轴显示范围从 -1 到 1
#
# # 添加标题和标签
# plt.xlabel('Time Windows', fontsize=18)
# plt.ylabel('Prediction Error', fontsize=18)
#
# # 添加图例
# plt.legend(fontsize=18)
# plt.xticks(fontsize=14)  # 放大 X 轴刻度字体大小
# plt.yticks(fontsize=14)  # 放大 Y 轴刻度字体大小
# # 显示图表
# plt.grid(True)
# plt.show()
#
# plt.plot(e33, label='Original Model', color='blue')
# plt.plot(ee33, label='Model With Structural Change', color='red', linestyle='--')
#
# plt.xlim(0, 300)  # x 轴显示范围从 0 到 10
# # plt.ylim(-0.5, 0.5)  # y 轴显示范围从 -1 到 1
#
# # 添加标题和标签
# plt.xlabel('Time Windows', fontsize=18)
# plt.ylabel('Prediction Error', fontsize=18)
#
# # 添加图例
# plt.legend(fontsize=18)
# plt.xticks(fontsize=14)  # 放大 X 轴刻度字体大小
# plt.yticks(fontsize=14)  # 放大 Y 轴刻度字体大小
# # 显示图表
# plt.grid(True)
# plt.show()

# from scipy import stats
#
#
# # 进行KS检验
# statistic, p_value = stats.ks_2samp(e0, ee0)
# # 输出结果
# print("KS statistic:", statistic)
# print("P-value:", p_value)
#
# # 进行KS检验
# statistic, p_value = stats.ks_2samp(e1, ee1)
# # 输出结果
# print("KS statistic:", statistic)
# print("P-value:", p_value)
#
# # 进行KS检验
# statistic, p_value = stats.ks_2samp(e2, ee2)
# # 输出结果
# print("KS statistic:", statistic)
# print("P-value:", p_value)
#
# # 进行KS检验
# statistic, p_value = stats.ks_2samp(e3, ee3)
# # 输出结果
# print("KS statistic:", statistic)
# print("P-value:", p_value)
#
# # 进行KS检验
# statistic, p_value = stats.ks_2samp(e03, ee03)
# # 输出结果
# print("KS statistic:", statistic)
# print("P-value:", p_value)
#
# # 进行KS检验
# statistic, p_value = stats.ks_2samp(e13, ee13)
# # 输出结果
# print("KS statistic:", statistic)
# print("P-value:", p_value)
#
# # 进行KS检验
# statistic, p_value = stats.ks_2samp(e23, ee23)
# # 输出结果
# print("KS statistic:", statistic)
# print("P-value:", p_value)
#
# # 进行KS检验
# statistic, p_value = stats.ks_2samp(e33, ee33)
# # 输出结果
# print("KS statistic:", statistic)
# print("P-value:", p_value)

abs_10 = [abs(x - y) for x, y in zip(e0, ee0)]
# 找到绝对值序列中的最大值
max_10 = max(abs_10)
print(max_10)

abs_11 = [abs(x - y) for x, y in zip(e1, ee1)]
# 找到绝对值序列中的最大值
max_11 = max(abs_11)
print(max_11)

abs_12 = [abs(x - y) for x, y in zip(e2, ee2)]
# 找到绝对值序列中的最大值
max_12 = max(abs_12)
print(max_12)

abs_13 = [abs(x - y) for x, y in zip(e3, ee3)]
# 找到绝对值序列中的最大值
max_13 = max(abs_13)
print(max_13)

abs_30 = [abs(x - y) for x, y in zip(e03, ee03)]
# 找到绝对值序列中的最大值
max_30 = max(abs_30)
print(max_30)

abs_31 = [abs(x - y) for x, y in zip(e13, ee13)]
# 找到绝对值序列中的最大值
max_31 = max(abs_31)
print(max_31)

abs_32 = [abs(x - y) for x, y in zip(e23, ee23)]
# 找到绝对值序列中的最大值
max_32 = max(abs_32)
print(max_32)

abs_33 = [abs(x - y) for x, y in zip(e33, ee33)]
# 找到绝对值序列中的最大值
max_33 = max(abs_33)
print(max_33)

print(1.36/pow(300, 1/2))

