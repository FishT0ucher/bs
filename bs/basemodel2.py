import numpy as np
import openpyxl
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import t
from torch.nn import Sequential
# from tensorflow.keras.layers import SimpleRNN, Dense
file_path = 'D:\\M4_1\\bs\\data3.xlsx'  # 请替换为您的文件路径
workbook = openpyxl.load_workbook(file_path)
sheet = workbook.active
mse_tree = []
mse_rf = []
mse_gbdt = []

# 获取第一列和第二列的数据
delay1 = [cell.value for cell in sheet['A']]
snr1 = [cell.value for cell in sheet['B']]
plr1 = [cell.value for cell in sheet['C']]
delay1 = np.array(delay1[1:]).reshape(-1, 1)
snr1 = np.array(snr1[1:]).reshape(-1, 1)
plr1 = np.array(plr1[1:]).reshape(-1, 1)

# 仅选择大于零的元素
positive_delay1 = delay1[delay1 > 0].reshape(-1, 1)
# 使用 MinMaxScaler 进行归一化
scaler1 = MinMaxScaler()
normalized_positive_delay1 = scaler1.fit_transform(positive_delay1)
# 将归一化后的结果填充回原始向量中
delay1[delay1 > 0] = normalized_positive_delay1.flatten()

# 仅选择大于零的元素
positive_snr1 = snr1[snr1 > 0].reshape(-1, 1)
# 使用 MinMaxScaler 进行归一化
scaler2 = MinMaxScaler()
normalized_positive_snr1 = scaler2.fit_transform(positive_snr1)
# 将归一化后的结果填充回原始向量中
snr1[snr1 > 0] = normalized_positive_snr1.flatten()

# 仅选择大于零的元素
positive_plr1 = plr1[plr1 > 0].reshape(-1, 1)
# 使用 MinMaxScaler 进行归一化
scaler3 = MinMaxScaler()
normalized_positive_plr1 = scaler3.fit_transform(positive_plr1)
# 将归一化后的结果填充回原始向量中
plr1[plr1 > 0] = normalized_positive_plr1.flatten()

# 计算分割索引
split_index = int(0.7 * len(snr1))
# 将 snr 向量分成前70%和后30%
snr1_train = snr1[:split_index]
snr1_test = snr1[split_index:]
delay1_train = delay1[:split_index]
delay1_test = delay1[split_index:]
plr1_train = plr1[:split_index]
plr1_test = plr1[split_index:]

# 创建特征矩阵 X 和标签向量 y
X_train = np.column_stack((snr1_train, delay1_train))
Y_train = plr1_train
X_test = np.column_stack((snr1_test, delay1_test))
Y_test = plr1_test
Y_pre_pos = Y_test[Y_test > 0].reshape(-1, 1)
Y_test[Y_test > 0] = scaler3.inverse_transform(Y_pre_pos).flatten()
# 创建回归树模型

mse_mlp = []

# 重复训练MLP模型
for i in range(10):
    reg_mlp = MLPRegressor(hidden_layer_sizes=(10, 10),  # 设置隐藏层神经元数量和层数
                           activation='relu',  # 使用ReLU作为激活函数
                           solver='adam',      # 使用Adam优化器
                           alpha=0.001,       # 正则化参数
                           batch_size='auto',  # 小批量梯度下降的批大小
                           learning_rate='constant',  # 使用固定的学习率
                           learning_rate_init=0.01,   # 初始学习率
                           max_iter=100,      # 最大迭代次数
                           tol=1e-4,           # 收敛容差
                           random_state=None)  # 随机种子

    reg_mlp.fit(X_train, Y_train)

    # 进行预测
    Y_pred = reg_mlp.predict(X_test)
    Y_pred = Y_pred.reshape(-1, 1)

    # 对预测值和真实值进行逆标准化（如果有必要）
    Y_pred_pos = Y_pred[Y_pred > 0].reshape(-1, 1)
    Y_pred[Y_pred > 0] = scaler3.inverse_transform(Y_pred_pos).flatten()

    # 计算均方误差并添加到列表中
    mse_test = np.mean((Y_pred - Y_test)**2)
    mse_mlp.append(mse_test)

# 将列表转换为 NumPy 数组
mse_mlp = np.array(mse_mlp)

# 计算均值
mean_mlp = np.mean(mse_mlp)

# 计算标准误差
std_error_mlp = np.std(mse_mlp, ddof=1) / np.sqrt(len(mse_mlp))

# 设置置信水平
confidence_level = 0.95

# 计算 t 分布的置信区间边界
t_value = t.ppf((1 + confidence_level) / 2, df=len(mse_mlp) - 1)

# 计算置信区间
lower_bound_mlp = mean_mlp - t_value * std_error_mlp
upper_bound_mlp = mean_mlp + t_value * std_error_mlp

#
# num_features = 2
# num_timesteps = 1
#
# # 调整输入形状
# X_train_reshaped = X_train.reshape(X_train.shape[0], num_timesteps, num_features)
# X_test_reshaped = X_test.reshape(X_test.shape[0], num_timesteps, num_features)
#
# # 创建一个空列表用于存储每次循环的均方误差
# mse_rnn = []
#
# # 重复训练 RNN 模型
# for i in range(10):
#     # 创建 RNN 模型
#     model = Sequential()
#     model.add(SimpleRNN(units=10, input_shape=(num_timesteps, num_features)))
#     model.add(Dense(1))  # 输出层，没有激活函数
#
#     # 编译模型
#     model.compile(optimizer='adam', loss='mse')
#
#     # 拟合模型
#     model.fit(X_train_reshaped, Y_train, epochs=10, batch_size=32, verbose=0)
#
#     # 进行预测
#     Y_pred = model.predict(X_test_reshaped)
#
#     # 对预测值和真实值进行逆标准化（如果有必要）
#     Y_pred_pos = Y_pred[Y_pred > 0].reshape(-1, 1)
#     Y_pred[Y_pred > 0] = scaler3.inverse_transform(Y_pred_pos).flatten()
#
#     # 计算均方误差并添加到列表中
#     mse_test = np.mean((Y_pred - Y_test)**2)
#     mse_rnn.append(mse_test)
#
# # 将列表转换为 NumPy 数组
# mse_rnn = np.array(mse_rnn)
#
# # 计算均值
# mean_rnn = np.mean(mse_rnn)
#
# # 计算标准误差
# std_error_rnn = np.std(mse_rnn, ddof=1) / np.sqrt(len(mse_rnn))
#
# # 设置置信水平
# confidence_level = 0.95
#
# # 计算 t 分布的置信区间边界
# t_value = t.ppf((1 + confidence_level) / 2, df=len(mse_rnn) - 1)
#
# # 计算置信区间
# lower_bound_rnn = mean_rnn - t_value * std_error_rnn
# upper_bound_rnn = mean_rnn + t_value * std_error_rnn

print(1)



