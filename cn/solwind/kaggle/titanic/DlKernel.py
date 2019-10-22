import tensorflow as tf
import numpy as np
import time
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 隐层节点数
HIDDEN_LAYER_POINT_NUM = 100
# 输入特征数
FEATURE_NUM = 11
# 输出结果数
RESULT_NUM = 2
# 迭代次数
TRAIN_STEP = 10000
# 学习率
TRAIN_LEARN_RATE = 0.9
# dropout时随机保留神经元的比例
DROPOUT_RATE = 0.9
# 是否使用DROPOUT层
BOL_DROPOUT = False
# 是否保存训练结果
BOL_SAVE_CHECKPOINT = True

# 日志保存路径
log_dir = "./eventlog"

labelEncoder = preprocessing.LabelEncoder()
oneHotEncoder = preprocessing.OneHotEncoder(sparse=False)

def load_dataset():
    # 读取训练数据
    data_train = pd.read_csv("./data/train.csv")
    target_train = data_train["Survived"]

    # 读取测试数据
    data_test = pd.read_csv("./data/test.csv")

    data_all = pd.concat([data_train, data_test], axis=0,sort=True)

    # 客舱分为有(1)无(0)两种
    data_train.loc[data_train.Cabin.notnull(), "Cabin"] = 1
    data_train.loc[data_train.Cabin.isnull(), "Cabin"] = 0

    data_test.loc[data_test.Cabin.notnull(), "Cabin"] = 1
    data_test.loc[data_test.Cabin.isnull(), "Cabin"] = 0

    data_all.loc[data_all.Cabin.notnull(), "Cabin"] = 1
    data_all.loc[data_all.Cabin.isnull(), "Cabin"] = 0

    # 补全测试集中的票价缺失
    df_fare_mean = data_all[["Pclass", "Fare"]].groupby(["Pclass"], as_index=True).mean()  # 根据Pclass计算均值
    # 获取缺失值对应Pclass下的均值进行填充
    data_test["Fare"].fillna(df_fare_mean.iloc[data_test.loc[data_test.Fare.isnull()]["Pclass"].values[0] - 1]["Fare"],
                             inplace=True)
    data_all["Fare"].fillna(df_fare_mean.iloc[data_all.loc[data_all.Fare.isnull()]["Pclass"].values[0] - 1]["Fare"],
                            inplace=True)
    # 预测并补全年龄数据
    # 抽取参与年龄计算的字段，使用全部数据进行预测
    age_data = data_all[["Age", "Pclass", "SibSp", "Parch", "Fare", "Cabin"]]

    age_null_data = age_data.loc[age_data.Age.isnull()]
    age_notnull_data = age_data.loc[age_data.Age.notnull()]
    X = age_notnull_data.values[:, 1:]
    y = age_notnull_data.values[:, 0]
    # 使用随机森林预测年龄
    rfr = RandomForestRegressor(n_estimators=400)
    rfr.fit(X, y)
    train_age_predict = rfr.predict(
        data_train[["Age", "Pclass", "SibSp", "Parch", "Fare", "Cabin"]].loc[data_train.Age.isnull()].values[:, 1:])
    data_train.loc[data_train.Age.isnull(), ["Age"]] = train_age_predict
    test_age_predict = rfr.predict(
        data_test[["Age", "Pclass", "SibSp", "Parch", "Fare", "Cabin"]].loc[data_test.Age.isnull()].values[:, 1:])
    data_test.loc[data_test.Age.isnull(), ["Age"]] = test_age_predict

    # 使用众数补全登船港口
    data_train["Embarked"].fillna(data_train["Embarked"].mode().iloc[0], inplace=True)

    # 特征处理
    # 性别oneHotEncoder
    oneHot_sex_train = pd.get_dummies(data_train['Sex'], prefix=data_train[['Sex']].columns[0])
    data_train = pd.concat([data_train, oneHot_sex_train], axis=1)
    oneHot_sex_test = pd.get_dummies(data_test['Sex'], prefix=data_test[['Sex']].columns[0])
    data_test = pd.concat([data_test, oneHot_sex_test], axis=1)
    # 港口oneHotEncoder
    oneHot_embarked_train = pd.get_dummies(data_train['Embarked'], prefix=data_train[['Embarked']].columns[0])
    data_train = pd.concat([data_train, oneHot_embarked_train], axis=1)
    oneHot_embarked_test = pd.get_dummies(data_test['Embarked'], prefix=data_test[['Embarked']].columns[0])
    data_test = pd.concat([data_test, oneHot_embarked_test], axis=1)
    # 是否生存oneHotEncoder
    oneHot_survied_train = pd.get_dummies(data_train['Survived'], prefix=data_train[['Survived']].columns[0])
    data_train = pd.concat([oneHot_survied_train,data_train], axis=1)

    drop_feature = ["Name", "Sex", "Ticket", "Embarked"]
    data_train.drop(drop_feature, axis=1, inplace=True)
    data_test.drop(drop_feature, axis=1, inplace=True)

    # 标准化
    scaler_feature = ["Age", "Fare"]
    scaler = RobustScaler()
    data_train[scaler_feature] = scaler.fit_transform(data_train[scaler_feature])
    data_test[scaler_feature] = scaler.fit_transform(data_test[scaler_feature])
    return data_train,data_test

data_train,data_test = load_dataset()
# print(data_train.head())
# print(data_train.values[:, 0:2])
# print("###################")

# 初始化权重
def init_weights(shape):
    return tf.Variable(tf.random.normal(shape, stddev=0.01))

# 初始化偏置量
def init_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

with tf.name_scope("input"):
    # 输入数据，n*10矩阵
    inputData = tf.round(tf.compat.v1.placeholder(tf.float32, shape=(None, FEATURE_NUM), name="inputData"))
    # 输入数据的结果(标签)n*2矩阵
    inputLabel = tf.compat.v1.placeholder(tf.float32, shape=(None, RESULT_NUM), name="inputLabel")

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope("summaries"):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.compat.v1.summary.scalar("mean", mean)

        # 计算参数的标准差
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.compat.v1.summary.scalar("stddev", stddev)
        tf.compat.v1.summary.scalar("max", tf.reduce_max(var))
        tf.compat.v1.summary.scalar("min", tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.compat.v1.summary.histogram("histogram", var)

# 创建神经网络层 @param:输入数据，本层节点数，下一层节点数，层名称，激活函数
def create_nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 设置命名空间
    with tf.name_scope(layer_name):
        # 初始化权重值
        with tf.name_scope("weights"):
            weights = init_weights([input_dim, output_dim])
            variable_summaries(weights)  # 记录权重信息

        # 初始化偏置量
        with tf.name_scope("biases"):
            biases = init_bias([output_dim])
            variable_summaries(biases)  # 记录偏置量信息

        # 执行Wx+b的线性计算，并且用直方图记录下来
        with tf.name_scope("linear_compute"):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.compat.v1.summary.histogram("linear", preactivate)

        # 将线性输出经过激励函数，并将输出也用直方图记录下来
        activations = act(preactivate, name="activation")
        tf.compat.v1.summary.histogram("activations", activations)

    # 返回激励层的最终输出
    return activations

# 创建隐层
layerHidden1 = create_nn_layer(inputData, FEATURE_NUM, HIDDEN_LAYER_POINT_NUM, "layerHidden1")

# 创建dropout层
with tf.name_scope('dropout'):
    keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_prob")
    tf.compat.v1.summary.scalar('dropout_keep_probability', keep_prob)
    layerDropout = tf.nn.dropout(layerHidden1, keep_prob)

layerToOut = layerHidden1

# 判断是否使用dropout层
if BOL_DROPOUT:
    layerToOut = layerDropout

# 创建输出层,tf.identity为恒等操作，具体逻辑可能会生成新的实例
layerOutput = create_nn_layer(layerToOut, HIDDEN_LAYER_POINT_NUM, RESULT_NUM, "layerOutput", act=tf.identity)

# 定义损失函数
with tf.name_scope("loss"):
    # 计算交叉熵损失（每个样本都会有一个损失）
    diff = tf.nn.softmax_cross_entropy_with_logits_v2(labels=inputLabel, logits=layerOutput)
    with tf.name_scope("total"):
        # 计算所有样本交叉熵损失的均值
        cross_entropy = tf.reduce_mean(diff)

tf.compat.v1.summary.scalar("loss", cross_entropy)

# 定义训练函数
with tf.name_scope("train"):
    train_op = tf.train.GradientDescentOptimizer(TRAIN_LEARN_RATE).minimize(cross_entropy)

# 定义预测准确率
with tf.name_scope("accuracy"):
    # 预测op，layerOutput为n*10矩阵，argmax为获取矩阵中某个纬度的最大值所在索引，也就是预测出的具体数字，第二个参数1代表每行计算最大值，0代表按列计算
    with tf.name_scope("correct_prediction"):
        # 分别将预测和真实的标签中取出最大值的索引，弱相同则返回1(true),不同则返回0(false)
        correct_prediction = tf.equal(tf.argmax(layerOutput, 1), tf.argmax(inputLabel, 1))
    with tf.name_scope("accuracy"):
        # 求均值即为准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="rate")
tf.compat.v1.summary.scalar("accuracy", accuracy * 100)


sess = tf.InteractiveSession()

# summaries合并
merged = tf.summary.merge_all()
# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
test_writer = tf.summary.FileWriter(log_dir + "/test")

# 运行初始化所有变量
tf.global_variables_initializer().run()

# 输入数据
def feed_dict(train):
    if train:
        # 训练
        X_train, X_test, y_train, y_test = train_test_split(data_train.values[:, 4:], data_train.values[:, 0:2], test_size=0.33, random_state=None)
        y_train = np.reshape(y_train,(y_train.shape[0],2))
        dropout = DROPOUT_RATE
    else:
        # 测试
        X_train = data_train.values[:, 4:]
        y_train = np.reshape(data_train.values[:, 0:2],(X_train.shape[0],2))
        dropout = 1.0
    return {inputData: X_train, inputLabel: y_train, keep_prob: dropout}


# 进行迭代训练
total_start_time = time.clock()
for i in range(TRAIN_STEP):
    start_time = time.clock()
    if i % 10 == 0:  # 记录测试集的summary与accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        end_time = time.clock()
        print("迭代[%s] 准确率:%.2f%% 耗时:%.0fms" % (i, acc * 100, (end_time - start_time) * 1000))
    else:  # 记录训练集的summary
        if i % 100 == 99:  # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_op],
                                  feed_dict=feed_dict(True),
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, "step%03d" % i)
            train_writer.add_summary(summary, i)
            # print("Adding run metadata for", i)
        else:  # Record a summary
            summary, _ = sess.run([merged, train_op], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
total_end_time = time.clock()
# 最终准确率
end_acc = sess.run(accuracy, feed_dict=feed_dict(False))
print("迭代次数[%s] 测试集准确率[%.2f%%] 学习率[%s] 隐层节点数[%s] Dropout[%s:%s] 总耗时[%.3fs]"
      % (i + 1, end_acc * 100, TRAIN_LEARN_RATE, HIDDEN_LAYER_POINT_NUM, BOL_DROPOUT, DROPOUT_RATE, (total_end_time - total_start_time)))
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
train_writer.close()
test_writer.close()

if BOL_SAVE_CHECKPOINT:
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/titanic")
