import numpy as np
import openpyxl
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import t

file_path = 'D:\\M4_1\\bs\\data2.xlsx'  # 请替换为您的文件路径
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


for i in range(10):
    reg_tree = DecisionTreeRegressor(max_depth=45)
    # 拟合模型
    reg_tree.fit(X_train, Y_train)

    # # 生成新的数据进行预测
    # new_data = np.array([[0.7, 0.4]])  # 例子，根据实际情况替换
    Y_pre = reg_tree.predict(X_test)
    Y_pre = Y_pre.reshape(-1, 1)

    Y_pre_pos = Y_pre[Y_pre > 0].reshape(-1, 1)
    Y_pre[Y_pre > 0] = scaler3.inverse_transform(Y_pre_pos).flatten()

    mse_test = np.mean((Y_pre - Y_test)**2)
    mse_tree.append(mse_test)
mse_tree = np.array(mse_tree).reshape(-1)

mean_tree = np.mean(mse_tree)
# 计算标准误差（Standard Error）
tree_std = np.std(mse_tree, ddof=1) / np.sqrt(len(mse_tree))
# 设置置信水平
confidence_level = 0.95
# 计算 t 分布的置信区间边界
t_value = t.ppf((1 + confidence_level) / 2, df=len(mse_tree)-1)

# 计算置信区间
lower_bound = mean_tree - t_value * tree_std
upper_bound = mean_tree + t_value * tree_std
# 创建一个空列表用于存储每次循环的均方误差
mse_forest = []

# 重复训练随机森林模型
for i in range(10):
    reg_forest = RandomForestRegressor(n_estimators=5, max_depth=15, bootstrap=True)  # 使用100棵树，最大深度为30
    reg_forest.fit(X_train, Y_train)

    # 进行预测
    Y_pred = reg_forest.predict(X_test)
    Y_pred = Y_pred.reshape(-1, 1)

    # 对预测值和真实值进行逆标准化（如果有必要）
    Y_pred_pos = Y_pred[Y_pred > 0].reshape(-1, 1)
    Y_pred[Y_pred > 0] = scaler3.inverse_transform(Y_pred_pos).flatten()

    # 计算均方误差并添加到列表中
    mse_test = np.mean((Y_pred - Y_test)**2)
    mse_forest.append(mse_test)

# 将列表转换为 NumPy 数组
mse_forest = np.array(mse_forest)

# 计算均值
mean_forest = np.mean(mse_forest)

# 计算标准误差
std_error_forest = np.std(mse_forest, ddof=1) / np.sqrt(len(mse_forest))

# 设置置信水平
confidence_level = 0.95

# 计算 t 分布的置信区间边界
t_value = t.ppf((1 + confidence_level) / 2, df=len(mse_forest) - 1)

# 计算置信区间
lower_bound_forest = mean_forest - t_value * std_error_forest
upper_bound_forest = mean_forest + t_value * std_error_forest


mse_gbdt = []

# 重复训练梯度提升决策树模型
for i in range(10):
    reg_gbdt = GradientBoostingRegressor(n_estimators=35,
                                         max_depth=35,   # 控制每棵树的最大深度
                                         learning_rate=0.1,  # 学习率
                                         subsample=0.75,
                                         min_samples_split=2,  # 自助采样的比例
                                         max_features='sqrt')  # 节点拆分所需的最小样本数

    reg_gbdt.fit(X_train, Y_train)

    # 进行预测
    Y_pred = reg_gbdt.predict(X_test)
    Y_pred = Y_pred.reshape(-1, 1)

    # 对预测值和真实值进行逆标准化（如果有必要）
    Y_pred_pos = Y_pred[Y_pred > 0].reshape(-1, 1)
    Y_pred[Y_pred > 0] = scaler3.inverse_transform(Y_pred_pos).flatten()

    # 计算均方误差并添加到列表中
    mse_test = np.mean((Y_pred - Y_test)**2)
    mse_gbdt.append(mse_test)

# 将列表转换为 NumPy 数组
mse_gbdt = np.array(mse_gbdt)

# 计算均值
mean_gbdt = np.mean(mse_gbdt)

# 计算标准误差
std_error_gbdt = np.std(mse_gbdt, ddof=1) / np.sqrt(len(mse_gbdt))

# 设置置信水平
confidence_level = 0.95

# 计算 t 分布的置信区间边界
t_value = t.ppf((1 + confidence_level) / 2, df=len(mse_gbdt) - 1)

# 计算置信区间
lower_bound_gbdt = mean_gbdt - t_value * std_error_gbdt
upper_bound_gbdt = mean_gbdt + t_value * std_error_gbdt
print(1)












# scaler = MinMaxScaler()
# normalized_delay1 = scaler.fit_transform(delay1)
# normalized_snr1 = scaler.fit_transform(snr1)
# normalized_plr1 = scaler.fit_transform(plr1)
#
# # 还原归一化后的向量
# pre_delay1 = scaler.inverse_transform(normalized_delay1)
# pre_snr1 = scaler.inverse_transform(normalized_snr1)
# pre_plr1 = scaler.inverse_transform(normalized_plr1)










