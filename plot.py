# 读取日志文件，实现 running_loss和mean-rmse可视化

# E:\Home_python\SGNet-main-1\SGNet-main\experiment\20241211210356-lr_0.0001-s_8-nyu_data-b_1\train-log
import re
import matplotlib.pyplot as plt
# 定义.log文件的路径
log_file_path = 'experiment/20251201184040-lr_0.0001-s_8-nyu_data-b_1/train.log'
# 初始化一个空列表来存储loss值
loss_values = []
rmse_values = []
# 定义一个正则表达式来匹配包含loss的行
loss_pattern = re.compile(r'running_loss:(\S+)')
rmse_pattern = re.compile(r'mean_rmse:(\S+)')

# 使用with语句打开文件
with open(log_file_path, 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 使用正则表达式匹配行
        match1 = loss_pattern.search(line)
        match2 = rmse_pattern.search(line)
        if match1:
            # 提取loss值
            loss_value = match1.group(1)
            # 将loss值转换为浮点数并添加到列表中
            loss_values.append(float(loss_value))
        if match2:
            # 提取rmse值
            rmse_value = match2.group(1)
            # 将rmse值转换为浮点数并添加到列表中
            rmse_values.append(float(rmse_value))

# 打印结果或进行其他处理
# print(loss_values)
print(len(loss_values))
# print(rmse_values)
print(len(rmse_values))

# 创建epoch值列表，从1到200
# epochs1 = list(range(1, 201))
epochs1 = list(range(1, len(loss_values)+1))
# epochs2 = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31-200]
epochs2 = list()
for i in range(15):
    epochs2.append(2 * i + 1)
# epochs2 = epochs2 + list(range(31, 201))
epochs2 = epochs2 + list(range(31, len(loss_values)+1))
print(epochs2)
print(len(epochs2)) # 185




# 绘制折线图
# running_loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs1, loss_values, marker='o', label='loss')  # marker='o'表示用圆圈标记每个点
# 设置图表标题和坐标轴标签
plt.title('running_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs2, rmse_values, marker='*', label='rmse_value')
# 设置图表标题和坐标轴标签
plt.title('mean_rmse')
plt.xlabel('Epochs')
plt.ylabel('rmse')
plt.legend()

# 显示网格
plt.grid(False)
# 显示图表
# plt.show()
# 保存图像
plt.savefig('loss_rmse_2.png')





