import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

# 隐层节点数
HIDDEN_LAYER_POTIN_NUM = 100
# 每个训练批次数据量
TRAIN_BATCH_NUM = 100
# 迭代次数
TRAIN_STEP = 10000
# 学习率
TRAIN_LEARN_RATE = 0.04
# dropout时随机保留神经元的比例
DROPOUT_RATE = 0.9
# 是否使用DROPOUT层
BOL_DROPOUT = True
# 是否保存训练结果
BOL_SAVE_CHECKPOINT = True

# 日志保存路径
log_dir = "./eventlog"

# 读取mnist数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# 初始化权重
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# 初始化偏置量
def init_bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


with tf.name_scope("input"):
    # 输入数据，n*784矩阵
    inputData = tf.round(tf.placeholder(tf.float32, shape=(None, 784), name="inputData"))
    # 输入数据的结果(标签)n*10矩阵
    inputLabel = tf.placeholder(tf.float32, shape=(None, 10), name="inputLabel")

# 将输入数据reshape成图片，保存入summary，用于tensorboard显示
with tf.name_scope("input_reshape"):
    image_shaped_input = tf.reshape(inputData, [-1, 28, 28, 1])
    tf.summary.image("input", image_shaped_input, 10)


# tensorboard显示用？？？？
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope("summaries"):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)

        # 计算参数的标准差
        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram("histogram", var)


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
            tf.summary.histogram("linear", preactivate)

        # 将线性输出经过激励函数，并将输出也用直方图记录下来
        activations = act(preactivate, name="activation")
        tf.summary.histogram("activations", activations)

    # 返回激励层的最终输出
    return activations


# 创建隐层
layerHidden1 = create_nn_layer(inputData, 784, HIDDEN_LAYER_POTIN_NUM, "layerHidden1")

# 创建dropout层
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    layerDropout = tf.nn.dropout(layerHidden1, keep_prob)

layerToOut = layerHidden1

# 判断是否使用dropout层
if BOL_DROPOUT:
    layerToOut = layerDropout

# 创建输出层,tf.identity为恒等操作，具体逻辑可能会生成新的实例
layerOutput = create_nn_layer(layerToOut, HIDDEN_LAYER_POTIN_NUM, 10, "layerOutput", act=tf.identity)

# 定义损失函数
with tf.name_scope("loss"):
    # 计算交叉熵损失（每个样本都会有一个损失）
    diff = tf.nn.softmax_cross_entropy_with_logits_v2(labels=inputLabel, logits=layerOutput)
    with tf.name_scope("total"):
        # 计算所有样本交叉熵损失的均值
        cross_entropy = tf.reduce_mean(diff)

tf.summary.scalar("loss", cross_entropy)

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
tf.summary.scalar("accuracy", accuracy * 100)

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
        image_data, result_label = mnist.train.next_batch(TRAIN_BATCH_NUM)
        print(type(image_data))
        print(type(result_label))
        dropout = DROPOUT_RATE
    else:
        # 测试
        image_data, result_label = mnist.test.images, mnist.test.labels
        dropout = 1.0
    return {inputData: image_data, inputLabel: result_label, keep_prob: dropout}


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
      % (i + 1, end_acc * 100, TRAIN_LEARN_RATE, HIDDEN_LAYER_POTIN_NUM, BOL_DROPOUT, DROPOUT_RATE,(total_end_time - total_start_time)))
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
train_writer.close()
test_writer.close()

if BOL_SAVE_CHECKPOINT:
    saver = tf.train.Saver()
    save_path = saver.save(sess, "./model/mnist")