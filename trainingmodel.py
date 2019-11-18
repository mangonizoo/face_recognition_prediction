import tensorflow as tf
import cv2
import numpy as np
#import os
from os import listdir
from os.path import isfile,join

global_step=tf.compat.v1.Variable(0,trainable=False,name='global_step')

def next_batch(x_data, y_data, batch_size):
    if (len(x_data) != len(y_data)):
        return None, None

    batch_mask = np.random.choice(len(x_data), batch_size)
    x_batch = np.array(x_data[batch_mask])
    y_batch = np.array(y_data[batch_mask])
    return x_batch, y_batch
#next_batch : 학습시킬 데이터(nparray)를 랜덤으로 가져옴(획일화 된 학습결과를 방지)
    
X = tf.compat.v1.placeholder(tf.float32, [None, 128, 128, 1])
Y = tf.compat.v1.placeholder(tf.float32, [None, 21])
keep_prob = tf.compat.v1.placeholder(tf.float32)
#X : 학습시킬 데이터(이미지 1장)이 들어갈 틀. / None:개수 지정안함. 128,128:128x128크기. 1:한 가지 색상
#Y: 학습 데이터를 구별할 라벨(총 21명)
#keep_prob:신경망 사용 퍼센테이지 지정 변수

#첫번째 layer(convolutional layer)
W1 = tf.compat.v1.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1))
B1 = tf.compat.v1.Variable(tf.truncated_normal([16],stddev=0.1))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(tf.add(L1,B1))
L1 = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#W1:가중치 설정. 3x3필터 지정/이미지를 3x3의 필터를 씌워 가중치 설정/뉴런의 개수 16개
#B1:편향 설정.
#필터의 이동속도를 1로 지정하여 한칸씩 이동, relu함수를 이용하여 결과값을 선형화
#max_pool2d를 이용하여 각 특징 별 최대값을 알아내어 해당 특징을 대표
#padding:결과값의 유실 방지
#아래의 5개 convolutional layer 모두 동일한 구조

#두번째 layer(convolutional layer)
W2 = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1))
B2 = tf.Variable(tf.truncated_normal([32],stddev=0.1))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(tf.add(L2,B2))
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#세번째 layer(convolutional layer)
W3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
B3 = tf.Variable(tf.truncated_normal([64],stddev=0.1))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(tf.add(L3,B3))
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#네번째 layer(convolutional layer)
W4 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
B4 = tf.Variable(tf.truncated_normal([128],stddev=0.1))
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(tf.add(L4,B4))
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#다섯번째 layer(convolutional layer)
W5 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
B5 = tf.Variable(tf.truncated_normal([256],stddev=0.1))
L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
L5 = tf.nn.relu(tf.add(L5,B5))
L5 = tf.nn.max_pool(L5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#여섯번째 layer(convolutional layer)
W6 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1))
B6 = tf.Variable(tf.truncated_normal([512],stddev=0.1))
L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
L6 = tf.nn.relu(tf.add(L6,B6))
L6 = tf.nn.max_pool(L6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#첫번째 Fully-connected layer
FC_W0 = tf.Variable(tf.truncated_normal([2 * 2 * 512, 1024], stddev=0.1))
FC_L0 = tf.reshape(L6, [-1, 2 * 2 * 512])
FC_B0 = tf.Variable(tf.truncated_normal([1024],stddev=0.1))
FC_L0 = tf.add(tf.matmul(FC_L0,FC_W0),FC_B0)
FC_L0 = tf.nn.relu(FC_L0)
#convolutional layer에서 추출된 특징 값을 기존의 뉴럴 네트워크에 넣어 분류
#FC_W0:가중치 설정
#FC_L0:직전의 convolutional layer를 1차원으로 reshape후 계산
#FC_B0:편향 설정
#아래의 fully-connected layer 구조 동일

FC_W1 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.1))
FC_B1 = tf.Variable(tf.truncated_normal([512],stddev=0.1))
FC_L1 = tf.add(tf.matmul(FC_L0,FC_W1),FC_B1)
FC_L1 = tf.nn.relu(FC_L1)

FC_W2 = tf.Variable(tf.truncated_normal([512, 256], stddev=0.1))
FC_B2 = tf.Variable(tf.truncated_normal([256],stddev=0.1))
FC_L2 = tf.add(tf.matmul(FC_L1,FC_W2),FC_B2)
FC_L2 = tf.nn.relu(FC_L2)
FC_L2 = tf.nn.dropout(FC_L2, keep_prob)
#위 계층과 동일하나 over fit을 막기위한 dropout 함수 사용(랜덤하게 뉴런을 꺼서 학습을 방해함)

#final layrt
FIN = tf.Variable(tf.truncated_normal([256, 21], stddev=0.1))
FIN_B = tf.Variable(tf.truncated_normal([21],stddev=0.1))
model = tf.add(tf.matmul(FC_L2,FIN),FIN_B)
#결과값을 라벨에 맞게 분류 및 모델 저장

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(cost)
#cost:모델에서 예측한 값과 기존의 라벨 값을 비교하여 저장
#optimizer:비용 값 최적화
#########
# 신경망 모델 학습
######
init = tf.compat.v1.global_variables_initializer()
sess = tf.compat.v1.Session()
sess.run(init)
#saver.save(sess,'./model/cnn.ckpt',global_step=global_step)

#################################################################################
#학습 데이터 읽어오기 및 라벨 지정
root_path='faces/'
pathes=[f for f in listdir(root_path)]
pathes=[x for x in pathes if x.find("new") != -1]
data_path=[]
img_files=[]
countimg=0

for i in range(len(pathes)):
    data_path.append(root_path+pathes[i]+'/')
    temp_data=[f for f in listdir(data_path[i]) if isfile(join(data_path[i],f))]
    img_files.append(temp_data)
    print(i," "+data_path[i])
    for j in range(len(img_files[i])):
#        print(img_files[i][j])
        countimg=countimg+1

Training_Data,Labels=[],[]

for i in range(len(pathes)):
    for j,files in enumerate(img_files[i]):
        image_path=data_path[i]+img_files[i][j]
        images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images,dtype=np.float32))
        Labels.append(i)

Labels=np.asarray(Labels,dtype=np.float32)
Training_Data=np.asarray(Training_Data,dtype=np.float32)

print(Training_Data[12])


Test_Labels=[]
for i in range(len(Labels)):
    temp=[]
    for x in range(21):
        if x==Labels[i]:
            temp.append(1)
        else:
            temp.append(0)
    Test_Labels.append(temp)
    
Test_Labels=np.array(Test_Labels)   
print(Test_Labels)
#################################################################################
saver=tf.train.Saver()

batch_size=100
total_batch = int(countimg/batch_size)
print(total_batch)
for epoch in range(1000):
    total_cost = 0
    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(Training_Data,Test_Labels,batch_size)
        batch_xs = batch_xs.reshape(-1, 128, 128, 1)

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          keep_prob: 0.7})
        total_cost += cost_val
    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    if (total_cost/total_batch)<0.01:
        saver.save(sess,'./testmodel/test2.ckpt')
        #모델 저장
        print("save complete")
        break

print('최적화 완료!')

#########
# 결과 확인
######


is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                        feed_dict={X: Training_Data.reshape(-1,128,128,1),
                                   Y: Test_Labels,
                                   keep_prob: 1}))


