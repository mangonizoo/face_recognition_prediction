# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 02:46:18 2019

@author: user
"""
import tensorflow as tf
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile,join
import datetime
import csv

namelist=['Adriana Lima','Alicia Keys','Angelina Jolie','Avril Lavigne'
          ,'Beyonce Knowles','Brad Pitt','Cameron Diaz',
          'Cate Blanchett','Charlize Theron', 'Colin Farrell','Colin Powell'
          ,'Daniel Radcliffe','David Beckham','Drew Barrymore',
          'Eva Mendes','George Clooney','Gwyneth Paltrow','Halle Berry',
          'Harrison Ford','Hugh Grant','Jennifer Aniston']

#예측을 위해 신경망 구조를 그대로 복사
global_step=tf.compat.v1.Variable(0,trainable=False,name='global_step')

X = tf.compat.v1.placeholder(tf.float32, [None, 128, 128, 1])
Y = tf.compat.v1.placeholder(tf.float32, [None, 21])
keep_prob = tf.compat.v1.placeholder(tf.float32)

W1 = tf.compat.v1.Variable(tf.truncated_normal([3, 3, 1, 16], stddev=0.1))
B1 = tf.compat.v1.Variable(tf.truncated_normal([16],stddev=0.1))
L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(tf.add(L1,B1))
L1 = tf.nn.max_pool2d(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.1))
B2 = tf.Variable(tf.truncated_normal([32],stddev=0.1))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(tf.add(L2,B2))
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
B3 = tf.Variable(tf.truncated_normal([64],stddev=0.1))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(tf.add(L3,B3))
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W4 = tf.Variable(tf.truncated_normal([3, 3, 64, 128], stddev=0.1))
B4 = tf.Variable(tf.truncated_normal([128],stddev=0.1))
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(tf.add(L4,B4))
L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W5 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
B5 = tf.Variable(tf.truncated_normal([256],stddev=0.1))
L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
L5 = tf.nn.relu(tf.add(L5,B5))
L5 = tf.nn.max_pool(L5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W6 = tf.Variable(tf.truncated_normal([3, 3, 256, 512], stddev=0.1))
B6 = tf.Variable(tf.truncated_normal([512],stddev=0.1))
L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
L6 = tf.nn.relu(tf.add(L6,B6))
L6 = tf.nn.max_pool(L6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

FC_W0 = tf.Variable(tf.truncated_normal([2 * 2 * 512, 1024], stddev=0.1))
FC_L0 = tf.reshape(L6, [-1, 2 * 2 * 512])
FC_B0 = tf.Variable(tf.truncated_normal([1024],stddev=0.1))
FC_L0 = tf.add(tf.matmul(FC_L0,FC_W0),FC_B0)
FC_L0 = tf.nn.relu(FC_L0)

FC_W1 = tf.Variable(tf.truncated_normal([1024, 512], stddev=0.1))
FC_B1 = tf.Variable(tf.truncated_normal([512],stddev=0.1))
FC_L1 = tf.add(tf.matmul(FC_L0,FC_W1),FC_B1)
FC_L1 = tf.nn.relu(FC_L1)

FC_W2 = tf.Variable(tf.truncated_normal([512, 256], stddev=0.1))
FC_B2 = tf.Variable(tf.truncated_normal([256],stddev=0.1))
FC_L2 = tf.add(tf.matmul(FC_L1,FC_W2),FC_B2)
FC_L2 = tf.nn.relu(FC_L2)
FC_L2 = tf.nn.dropout(FC_L2, keep_prob)

FIN = tf.Variable(tf.truncated_normal([256, 21], stddev=0.1))
FIN_B = tf.Variable(tf.truncated_normal([21],stddev=0.1))
model = tf.add(tf.matmul(FC_L2,FIN),FIN_B)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.compat.v1.train.AdamOptimizer(0.001).minimize(cost)

sess=tf.compat.v1.Session()
saver=tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

#괄호 안의 이름과 일치하는 모델 파일의 존재를 확인하기위한 변수
ckpt=tf.compat.v1.train.get_checkpoint_state('./testmodel')

##주의 :: 모델 불러올 때, 학습 시킨 후 커널 초기화하고 시작해야함 -> 모델을 생성한 후 데이터가 커널에 남아있음
if ckpt and tf.compat.v1.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess,ckpt.model_checkpoint_path)
    print("model exists")
    #모델이 존재하면 불러옴
else:
    sess.run(tf.compat.v1.global_variable_initializer())

#################################################################################
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#얼굴추출을 위한 api

all_uk=[]
data_log=[]

#인물예측 시 해당되는 인물이 없을 경우를 판단하기 위한 배열
for i in range(21):
    all_uk.append(-1)
all_uk=np.asarray(all_uk,dtype=np.float32)

#예측한 값을 저장하기위한 함수
def find(prediction,count):
    templist=[]
    checklist=[]
#    ar_length=prediction.size/21
    for i in range(count):
        templist2=[]
        for j in range(21):
            if prediction[i][j] < 0.9:
                templist2.append(-1)
            else:
                templist2.append(prediction[i][j])
        
        templist2=np.asarray(templist2,dtype=np.float32)
        templist.append(templist2)
        if np.array_equal(templist2,all_uk):
            checklist.append(False)
        else:
            checklist.append(True)
    templist=np.asarray(templist,dtype=np.float32)
    return templist,checklist

#영상의 주소를 입력
print(">>please input the video's path")
a=input()
videopath=a

video=cv2.VideoCapture(videopath)

#영상에서 얼굴 추출 및 예측
if video.isOpened() == False:
    print("can't load")
    exit()

while(video.isOpened()):
    ret, frame = video.read()
    if ret:#영상이 재생되고 있는 경우
        grayframe=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#영상을 흑백처리
        testdata=[]
        im_resized=[]
        count=0
        try:
            faces=face_classifier.detectMultiScale(grayframe,1.3,5)#영상에서 얼굴 추출
#        print(faces)
            for (x,y,w,h) in faces:#추출 된 얼굴을 대상으로 예측 진행
                count=count+1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                im_cropped=frame[y-int(h/4):y+h+int(h/4),x-int(w/4):x+w+int(w/4)]
                im_resized=cv2.cvtColor(im_cropped,cv2.COLOR_BGR2GRAY)
                im_resized=cv2.resize(im_resized,(128,128),None,fx=1,fy=1,interpolation=cv2.INTER_AREA)
                im_resized=np.asarray(im_resized,dtype=np.float32)
                testdata.append(im_resized)
                #추출된 얼굴을 예측하기 위한 자료형으로 resizing 및 reshape
            if count>=1:#인물이 1명 이상일 경우 예측 시작
                checklist=[]
                testdata=np.asarray(testdata,dtype=np.float32)
                testmodel=tf.nn.softmax(model)
                tt= sess.run(testmodel,feed_dict={X:testdata.reshape(-1,128,128,1),keep_prob:1.0})
                tt,checklist=find(tt,count)
                tt=np.argmax(tt,1)#예측된 값 중 한명당 가장 유사한 인물로 지정
                temp=count
                for (x,y,w,h) in faces:#얼굴마다
                    ar_index=count-temp
                    if checklist[ar_index]==False:#예측정보가 없을 경우 unknown
                        cv2.putText(frame,"unknown",(x,y),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
                    else:#예측 성공시 해당 라벨에 맞는 인물의 이름을 얼굴위에 출력
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                        cv2.putText(frame,namelist[tt[ar_index]],(x,y),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2)
                        data_log.append([tt[ar_index],datetime.datetime.now()])#해당 인물의 라벨 및 시각 저장
                    temp=temp-1
        except:
            continue
        cv2.imshow("face",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
#cv2.waitKey(0)
video.release()
#cv2.waitKey(0)
cv2.destroyWindow("face")
#저장된 인물,시각 정보를 csv파일로 저장
file=open('data.csv','w',encoding='utf-8',newline='')
csvfile=csv.writer(file)
for row in data_log:
    csvfile.writerow(row)
file.close()
print("csv file save")

#################################################################################
#testmodel=model
#testmodel=tf.nn.softmax(testmodel)
#
#prediction=tf.math.argmax(model,1)
##testmodel=tf.cast(testmodel,tf.float16)
##testmodel=tf.nn.softmax(testmodel)
#testmodel=tf.cast(testmodel>0.8,tf.float16)
#testmodel=tf.math.argmax(testmodel,1)
#
#print('테스트:',sess.run(testmodel,
#                      feed_dict={X:Validate_Data.reshape(-1,128,128,1),keep_prob:1.0},
#                      ))
#print(tf.size(testmodel))
#print(testmodel[1])
#target=tf.math.argmax(Y,1)
##testtarget=[]
##testtarget=np.asarray(target,dtype=np.int)
##for i in range(len(testtarget)):
##    print(testtarget[i])
#print('예측값:',sess.run(prediction,
#                      feed_dict={X:Validate_Data.reshape(-1,128,128,1),keep_prob:1.0},
#                      ))
#print('실제값:',sess.run(target,
#                      feed_dict={Y:TL}))
#
#is_correct=tf.compat.v1.equal(prediction,target)
#accuracy=tf.compat.v1.reduce_mean(tf.compat.v1.cast(is_correct,tf.float32))
#print('정확도: %.2f' % sess.run(accuracy*100,
#                             feed_dict={X:Validate_Data.reshape(-1,128,128,1),Y:TL,keep_prob:1.0}))




