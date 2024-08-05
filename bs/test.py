# import torch
# import torch.nn as nn
# from matplotlib import pyplot as plt
# from torch.optim import Adam
# import numpy as np
# import openpyxl
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.neural_network import MLPRegressor
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.tree import DecisionTreeRegressor
# from scipy.stats import t
# import torch
# import torch.nn as nn
# from torch import optim
# import torch.nn.functional as F
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error, r2_score
# mse_tree = []
# mse_rf = []
# mse_gbdt = []
#
# file_path1 = 'D:\\M4_1\\bs\\data1.xlsx'  # 请替换为您的文件路径
# workbook1 = openpyxl.load_workbook(file_path1)
# sheet1 = workbook1.active
#
# file_path2 = 'D:\\M4_1\\bs\\data2.xlsx'  # 请替换为您的文件路径
# workbook2 = openpyxl.load_workbook(file_path2)
# sheet2 = workbook2.active
#
# file_path3 = 'D:\\M4_1\\bs\\data3.xlsx'  # 请替换为您的文件路径
# workbook3 = openpyxl.load_workbook(file_path3)
# sheet3 = workbook3.active
#
# delay1 = [cell.value for cell in sheet1['A']]
# snr1 = [cell.value for cell in sheet1['B']]
# plr1 = [cell.value for cell in sheet1['C']]
# delay1 = np.array(delay1[1:]).reshape(-1, 1)
# snr1 = np.array(snr1[1:]).reshape(-1, 1)
# plr1 = np.array(plr1[1:]).reshape(-1, 1)
#
# delay2 = [cell.value for cell in sheet2['A']]
# snr2 = [cell.value for cell in sheet2['B']]
# plr2 = [cell.value for cell in sheet2['C']]
# delay2 = np.array(delay2[1:]).reshape(-1, 1)
# snr2 = np.array(snr2[1:]).reshape(-1, 1)
# plr2 = np.array(plr2[1:]).reshape(-1, 1)
#
# delay3 = [cell.value for cell in sheet3['A']]
# snr3 = [cell.value for cell in sheet3['B']]
# plr3 = [cell.value for cell in sheet3['C']]
# plr0 = [cell.value for cell in sheet3['D']]
# delay3 = np.array(delay3[1:]).reshape(-1, 1)
# snr3 = np.array(snr3[1:]).reshape(-1, 1)
# plr3 = np.array(plr3[1:]).reshape(-1, 1)
# plr0 = np.array(plr0[1:]).reshape(-1, 1)
#
#
# # 仅选择大于零的元素
# positive_delay1 = delay1[delay1 > 0].reshape(-1, 1)
# # 使用 MinMaxScaler 进行归一化
# scaler_delay1 = MinMaxScaler()
# normalized_positive_delay1 = scaler_delay1.fit_transform(positive_delay1)
# # 将归一化后的结果填充回原始向量中
# delay1[delay1 > 0] = normalized_positive_delay1.flatten()
#
# # 仅选择大于零的元素
# positive_snr1 = snr1[snr1 > 0].reshape(-1, 1)
# # 使用 MinMaxScaler 进行归一化
# scaler_snr1 = MinMaxScaler()
# normalized_positive_snr1 = scaler_snr1.fit_transform(positive_snr1)
# # 将归一化后的结果填充回原始向量中
# snr1[snr1 > 0] = normalized_positive_snr1.flatten()
#
# # 仅选择大于零的元素
# positive_plr1 = plr1[plr1 > 0].reshape(-1, 1)
# # 使用 MinMaxScaler 进行归一化
# scaler_plr1 = MinMaxScaler()
# normalized_positive_plr1 = scaler_plr1.fit_transform(positive_plr1)
# # 将归一化后的结果填充回原始向量中
# plr1[plr1 > 0] = normalized_positive_plr1.flatten()
#
#
# # 仅选择大于零的元素
# positive_delay2 = delay2[delay2 > 0].reshape(-1, 1)
# # 使用 MinMaxScaler 进行归一化
# scaler_delay2 = MinMaxScaler()
# normalized_positive_delay2 = scaler_delay2.fit_transform(positive_delay2)
# # 将归一化后的结果填充回原始向量中
# delay2[delay2 > 0] = normalized_positive_delay2.flatten()
#
# # 仅选择大于零的元素
# positive_snr2 = snr2[snr2 > 0].reshape(-1, 1)
# # 使用 MinMaxScaler 进行归一化
# scaler_snr2 = MinMaxScaler()
# normalized_positive_snr2 = scaler_snr2.fit_transform(positive_snr2)
# # 将归一化后的结果填充回原始向量中
# snr2[snr2 > 0] = normalized_positive_snr2.flatten()
#
# # 仅选择大于零的元素
# positive_plr2 = plr2[plr2 > 0].reshape(-1, 1)
# # 使用 MinMaxScaler 进行归一化
# scaler_plr2 = MinMaxScaler()
# normalized_positive_plr2 = scaler_plr2.fit_transform(positive_plr2)
# # 将归一化后的结果填充回原始向量中
# plr2[plr2 > 0] = normalized_positive_plr2.flatten()
#
#
# # 仅选择大于零的元素
# positive_delay3 = delay3[delay3 > 0].reshape(-1, 1)
# # 使用 MinMaxScaler 进行归一化
# scaler_delay3 = MinMaxScaler()
# normalized_positive_delay3 = scaler_delay3.fit_transform(positive_delay3)
# # 将归一化后的结果填充回原始向量中
# delay3[delay3 > 0] = normalized_positive_delay3.flatten()
#
# # 仅选择大于零的元素
# positive_snr3 = snr3[snr3 > 0].reshape(-1, 1)
# # 使用 MinMaxScaler 进行归一化
# scaler_snr3 = MinMaxScaler()
# normalized_positive_snr3 = scaler_snr3.fit_transform(positive_snr3)
# # 将归一化后的结果填充回原始向量中
# snr3[snr3 > 0] = normalized_positive_snr3.flatten()
#
# # 仅选择大于零的元素
# positive_plr3 = plr3[plr3 > 0].reshape(-1, 1)
# # 使用 MinMaxScaler 进行归一化
# scaler_plr3 = MinMaxScaler()
# normalized_positive_plr3 = scaler_plr3.fit_transform(positive_plr3)
# # 将归一化后的结果填充回原始向量中
# plr3[plr3 > 0] = normalized_positive_plr3.flatten()
#
# # 仅选择大于零的元素
# positive_plr0 = plr0[plr0 > 0].reshape(-1, 1)
# # 使用 MinMaxScaler 进行归一化
# scaler_plr0 = MinMaxScaler()
# normalized_positive_plr0 = scaler_plr0.fit_transform(positive_plr0)
# # 将归一化后的结果填充回原始向量中
# plr0[plr0 > 0] = normalized_positive_plr0.flatten()
#
#
# # 计算分割索引
# split_index = int(0.7 * len(snr1))
# # 将 snr 向量分成前70%和后30%
# snr1_train = snr1[:split_index]
# snr1_test = snr1[split_index:]
# delay1_train = delay1[:split_index]
# delay1_test = delay1[split_index:]
# plr1_train = plr1[:split_index]
# plr1_test = plr1[split_index:]
#
#
# snr2_train = snr2[:split_index]
# snr2_test = snr2[split_index:]
# delay2_train = delay2[:split_index]
# delay2_test = delay2[split_index:]
# plr2_train = plr2[:split_index]
# plr2_test = plr2[split_index:]
#
#
# snr3_train = snr3[:split_index]
# snr3_test = snr3[split_index:]
# delay3_train = delay3[:split_index]
# delay3_test = delay3[split_index:]
# plr3_train = plr3[:split_index]
# plr3_test = plr3[split_index:]
#
#
# plr0_train = plr0[:split_index]
# plr0_test = plr0[split_index:]
#
#
# # 创建特征矩阵 X 和标签向量 y
# X_train1 = np.column_stack((snr1_train, delay1_train))
# Y_train1 = plr1_train
# X_test1 = np.column_stack((snr1_test, delay1_test))
# Y_test1 = plr1_test
# Y_pre_pos1 = Y_test1[Y_test1 > 0].reshape(-1, 1)
# Y_test1[Y_test1 > 0] = scaler_plr1.inverse_transform(Y_pre_pos1).flatten()
#
#
# X_train2 = np.column_stack((snr2_train, delay2_train))
# Y_train2 = plr2_train
# X_test2 = np.column_stack((snr2_test, delay2_test))
# Y_test2 = plr2_test
# Y_pre_pos2 = Y_test2[Y_test2 > 0].reshape(-1, 1)
# Y_test2[Y_test2 > 0] = scaler_plr2.inverse_transform(Y_pre_pos2).flatten()
#
#
# X_train3 = np.column_stack((snr3_train, delay3_train))
# Y_train3 = plr3_train
# X_test3 = np.column_stack((snr3_test, delay3_test))
# Y_test3 = plr3_test
# Y_pre_pos3 = Y_test3[Y_test3 > 0].reshape(-1, 1)
# Y_test3[Y_test3 > 0] = scaler_plr3.inverse_transform(Y_pre_pos3).flatten()
#
# Y_train0 = plr0_train
# Y_test0 = plr0_test
# Y_pre_pos0 = Y_test0[Y_test0 > 0].reshape(-1, 1)
# Y_test0[Y_test0 > 0] = scaler_plr0.inverse_transform(Y_pre_pos0).flatten()
#
# reg_forest1 = RandomForestRegressor(n_estimators=25, max_depth=15, bootstrap=True)  # 使用100棵树，最大深度为30
# reg_forest1.fit(X_train1, Y_train1)
#
# # 进行预测
# Y_T1 = reg_forest1.predict(X_test1)
# Y_T1 = Y_T1.reshape(-1, 1)
#
# reg_forest2 = RandomForestRegressor(n_estimators=25, max_depth=15, bootstrap=True)  # 使用100棵树，最大深度为30
# reg_forest2.fit(X_train2, Y_train2)
#
# # 进行预测
# Y_T2 = reg_forest2.predict(X_test2)
# Y_T2 = Y_T2.reshape(-1, 1)
#
# reg_forest3 = RandomForestRegressor(n_estimators=25, max_depth=15, bootstrap=True)  # 使用100棵树，最大深度为30
# reg_forest3.fit(X_train3, Y_train3)
#
# # 进行预测
# Y_T3 = reg_forest3.predict(X_test3)
# Y_T3 = Y_T3.reshape(-1, 1)
#
# reg_forest0 = RandomForestRegressor(n_estimators=25, max_depth=15, bootstrap=True)  # 使用100棵树，最大深度为30
# reg_forest0.fit(X_train3, Y_train0)
#
# # 进行预测
# Y_T0 = reg_forest0.predict(X_test3)
# Y_T0 = Y_T0.reshape(-1, 1)
#
# class Transformer_Encoder(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(Transformer_Encoder, self).__init__()
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=1)
#         self.hidden_size = hidden_size
#         self.lstm = nn.LSTM(input_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.tanh = nn.ELU(alpha=50)
#
#     def forward(self, x):
#         # 前向传播
#         out = self.encoder_layer(x)
#         out = out.view(1, -1)
#         out = self.fc(out)
#         out = self.tanh(out)
#         return out
#
#
# reg_mlp1 = MLPRegressor(hidden_layer_sizes=(10, 10),  # 设置隐藏层神经元数量和层数
#                         activation='relu',  # 使用ReLU作为激活函数
#                         solver='adam',  # 使用Adam优化器
#                         alpha=0.001,  # 正则化参数
#                         batch_size='auto',  # 小批量梯度下降的批大小
#                         learning_rate='constant',  # 使用固定的学习率
#                         learning_rate_init=0.01,  # 初始学习率
#                         max_iter=50,  # 最大迭代次数
#                         tol=1e-4,  # 收敛容差
#                         random_state=None)  # 随机种子
#
# reg_mlp1.fit(X_test1, Y_test1)
#
# # 进行预测
# Y_F1 = reg_mlp1.predict(X_test1)
# Y_F1 = Y_F1.reshape(-1, 1)
#
#
# reg_mlp2 = MLPRegressor(hidden_layer_sizes=(10, 10),  # 设置隐藏层神经元数量和层数
#                         activation='relu',  # 使用ReLU作为激活函数
#                         solver='adam',  # 使用Adam优化器
#                         alpha=0.001,  # 正则化参数
#                         batch_size='auto',  # 小批量梯度下降的批大小
#                         learning_rate='constant',  # 使用固定的学习率
#                         learning_rate_init=0.01,  # 初始学习率
#                         max_iter=50,  # 最大迭代次数
#                         tol=1e-4,  # 收敛容差
#                         random_state=None)  # 随机种子
#
# reg_mlp2.fit(X_test2, Y_test2)
#
# # 进行预测
# Y_F2 = reg_mlp2.predict(X_test2)
# Y_F2 = Y_F2.reshape(-1, 1)
#
#
# reg_mlp3 = MLPRegressor(hidden_layer_sizes=(10, 10),  # 设置隐藏层神经元数量和层数
#                         activation='relu',  # 使用ReLU作为激活函数
#                         solver='adam',  # 使用Adam优化器
#                         alpha=0.001,  # 正则化参数
#                         batch_size='auto',  # 小批量梯度下降的批大小
#                         learning_rate='constant',  # 使用固定的学习率
#                         learning_rate_init=0.01,  # 初始学习率
#                         max_iter=50,  # 最大迭代次数
#                         tol=1e-4,  # 收敛容差
#                         random_state=None)  # 随机种子
#
# reg_mlp3.fit(X_test3, Y_test3)
#
# # 进行预测
# Y_F3 = reg_mlp3.predict(X_test3)
# Y_F3 = Y_F3.reshape(-1, 1)
#
# reg_mlp0 = MLPRegressor(hidden_layer_sizes=(10, 10),  # 设置隐藏层神经元数量和层数
#                         activation='relu',  # 使用ReLU作为激活函数
#                         solver='adam',  # 使用Adam优化器
#                         alpha=0.001,  # 正则化参数
#                         batch_size='auto',  # 小批量梯度下降的批大小
#                         learning_rate='constant',  # 使用固定的学习率
#                         learning_rate_init=0.01,  # 初始学习率
#                         max_iter=50,  # 最大迭代次数
#                         tol=1e-4,  # 收敛容差
#                         random_state=None)  # 随机种子
#
# reg_mlp0.fit(X_test3, Y_test0)
#
# # 进行预测
# Y_F0 = reg_mlp0.predict(X_test3)
# Y_F0 = Y_F0.reshape(-1, 1)
#
#
# # plt.plot(Y_T0, label='Random Forest', color='blue', linestyle='-.')
# plt.plot(Y_F0, label='Transformer Encoder', color='yellow')
# # 绘制真实值 y 的折线图
# plt.plot(Y_test0, label='True Values', color='red', linestyle='--')
# #
# # zero_indices = np.where(y_mean1 == 0.3)[0]
# # c1 = zero_indices
# # plt.scatter(zero_indices, y_mean1[zero_indices], color='green', marker='x', s=200, linewidths=4, label='No Data Transmission')
# plt.xlim(0, 300)  # x 轴显示范围从 0 到 10
# # plt.ylim(0.3, 0.5)  # y 轴显示范围从 -1 到 1
#
# # 添加标题和标签
# plt.xlabel('Time Windows', fontsize=18)
# plt.ylabel('Packet Loss Rate', fontsize=18)
#
# # 添加图例
# plt.legend(fontsize=18)
# plt.xticks(fontsize=14)  # 放大 X 轴刻度字体大小
# plt.yticks(fontsize=14)  # 放大 Y 轴刻度字体大小
# # 显示图表
# plt.grid(True)
# plt.show()







# # 创建模型实例
# input_size = 1
# hidden_size = 2
# output_size = 1
# Y_R1 = []
#
# T1 = Transformer_Encoder(input_size, hidden_size, output_size)
# # 定义损失函数和优化器
# criterion1 = nn.MSELoss()
# optimizer1 = optim.Adam(T1.parameters(), lr=0.01)
#
# # 准备训练数据
#
# x_train_r1 = torch.tensor(X_train1, dtype=torch.float32).view(-1, 2)  # 输入维度为 (x, 2)
# y_train_r1 = torch.tensor(Y_train1, dtype=torch.float32).view(-1, 1)            # 输出维度为 (x, 1)
#
# # 训练模型
# num_epochs = 5
# for epoch in range(num_epochs):
#     # 前向传播
#     for batchnum in range(len(x_train_r1)):
#         x_in = x_train_r1[batchnum]
#         outputs = T1(x_in.view(len(x_in), 1, -1))  # 修改这里的输入维度
#         if epoch == num_epochs-1:
#             Y_R1.append(outputs.detach().numpy())
#         # 计算损失
#         loss = criterion1(outputs, y_train_r1[batchnum])
#
#         # 反向传播和优化
#         optimizer1.zero_grad()
#         loss.backward()
#         optimizer1.step()
#
#     # # 打印训练信息
#     # if (epoch+1) % 2 == 0:
#     #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# Y_R1 = np.array(Y_R1).reshape(-1, 1)


import numpy as np

def change_sign(array):
    random_array = np.random.rand(*array.shape)  # 生成与输入数组形状相同的随机数组
    change_mask = random_array < 0.3  # 生成一个布尔掩码，确定是否改变符号
    result = np.where(change_mask, -array, array)  # 根据掩码改变符号
    return result

# 生成一个示例数组
input_array = np.array([1, 2, -3, 4, -5])

# 对数组进行处理
output_array = change_sign(input_array)

print("原始数组：", input_array)
print("处理后的数组：", output_array)