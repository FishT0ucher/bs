import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats import t

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

y01 = np.floor(10*(y01 + 3*y0)/4)
y02 = np.floor(10*(y02 + 3*y0)/4)
y03 = np.floor(10*(y03 + 3*y0)/4)
y04 = np.floor(10*(y04 + 3*y0)/4)
y05 = np.floor(10*(y05 + 3*y0)/4)

y11 = np.floor(10*(y11 + y1)/2)
y12 = np.floor(10*(y12 + y1)/2)
y13 = np.floor(10*(y13 + y1)/2)
y14 = np.floor(10*(y14 + y1)/2)
y15 = np.floor(10*(y15 + y1)/2)

y21 = np.floor(10*(y21 + y2)/2)
y22 = np.floor(10*(y22 + y2)/2)
y23 = np.floor(10*(y23 + y2)/2)
y24 = np.floor(10*(y24 + y2)/2)
y25 = np.floor(10*(y25 + y2)/2)

y31 = np.floor(10*(y31 + 2*y3)/3)
y32 = np.floor(10*(y32 + 2*y3)/3)
y33 = np.floor(10*(y33 + 2*y3)/3)
y34 = np.floor(10*(y34 + 2*y3)/3)
y35 = np.floor(10*(y35 + 2*y3)/3)

y0 = np.floor(10*y0)
y1 = np.floor(10*y1)
y2 = np.floor(10*y2)
y3 = np.floor(10*y3)

equal_elements = np.equal(y0, y01)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y01r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y0, y02)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y02r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y0, y03)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y03r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y0, y04)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y04r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y0, y05)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y05r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y1, y11)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y11r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y1, y12)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y12r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y1, y13)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y13r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y1, y14)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y14r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y1, y15)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y15r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y2, y21)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y21r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y2, y22)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y22r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y2, y23)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y23r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y2, y24)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y24r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y2, y25)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y25r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y3, y31)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y31r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y3, y32)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y32r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y3, y33)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y33r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y3, y34)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y34r = num_equal_elements / len(y0) * 100

equal_elements = np.equal(y3, y35)
# 计算相等元素的数量
num_equal_elements = np.sum(equal_elements)
# 计算相同元素的比例
y35r = num_equal_elements / len(y0) * 100

v0 = [y01r, y02r, y03r, y04r, y05r]
v1 = [y11r, y12r, y13r, y14r, y15r]
v2 = [y21r, y22r, y23r, y24r, y25r]
v3 = [y31r, y32r, y33r, y34r, y35r]

mean_v0 = np.mean(v0)
mean_v1 = np.mean(v1)
mean_v2 = np.mean(v2)
mean_v3 = np.mean(v3)

# 计算标准误差
std_error_v0 = np.std(v0, ddof=1) / np.sqrt(len(v0))
std_error_v1 = np.std(v1, ddof=1) / np.sqrt(len(v1))
std_error_v2 = np.std(v2, ddof=1) / np.sqrt(len(v2))
std_error_v3 = np.std(v3, ddof=1) / np.sqrt(len(v3))

# 设置置信水平
confidence_level = 0.95

# 计算 t 分布的置信区间边界
t_value = t.ppf((1 + confidence_level) / 2, df=len(v1) - 1)

# 计算置信区间
lower_bound_v0 = mean_v0 - t_value * std_error_v0
upper_bound_v0 = mean_v0 + t_value * std_error_v0

lower_bound_v1 = mean_v1 - t_value * std_error_v1
upper_bound_v1 = mean_v1 + t_value * std_error_v1

lower_bound_v2 = mean_v2 - t_value * std_error_v2
upper_bound_v2 = mean_v2 + t_value * std_error_v2

lower_bound_v3 = mean_v3 - t_value * std_error_v3
upper_bound_v3 = mean_v3 + t_value * std_error_v3

error_v0 = (upper_bound_v0 - lower_bound_v0) / 2
error_v1 = (upper_bound_v1 - lower_bound_v1) / 2
error_v2 = (upper_bound_v2 - lower_bound_v2) / 2
error_v3 = (upper_bound_v3 - lower_bound_v3) / 2

# 设置条形图的位置和宽度
bar_width = 0.35
index = np.arange(4)

# 绘制条形图
plt.bar(index, [mean_v0, mean_v1, mean_v2, mean_v3], yerr=[[error_v0, error_v1, error_v2, error_v3],
                                                           [error_v0, error_v1, error_v2, error_v3]], capsize=20, label='95% Confidence Interval')

# 添加标签和标题
plt.xlabel('Channel', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.xticks(index, ['SD', '1', '2', '3'], fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=18)
plt.grid(True)
# 显示图形
plt.show()



#
# # 添加标题和标签
# plt.xlabel('Time Windows', fontsize=18)
# plt.ylabel('Packet Loss Rate', fontsize=18)
#
# # 添加图例
# plt.legend(fontsize=18)
# plt.xticks(fontsize=16)  # 放大 X 轴刻度字体大小
# plt.yticks(fontsize=16)  # 放大 Y 轴刻度字体大小
# # 显示图表
# plt.grid(True)
# plt.show()

























print(0)