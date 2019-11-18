
import csv
import numpy as np
import pymysql

pymysql.converters.encoders[np.int64] = pymysql.converters.escape_float
pymysql.converters.conversions = pymysql.converters.encoders.copy()
pymysql.converters.conversions.update(pymysql.converters.decoders)

#database연동을 위한 connector 및 cursor
conn = pymysql.connect(host = "localhost", user = "root", password = "1233" ,db = "new_schema")
curs = conn.cursor()

namelist=['Adriana Lima','Alicia Keys','Angelina Jolie','Avril Lavigne'
          ,'Beyonce Knowles','Brad Pitt','Cameron Diaz',
          'Cate Blanchett','Charlize Theron', 'Colin Farrell','Colin Powell'
          ,'Daniel Radcliffe','David Beckham','Drew Barrymore',
          'Eva Mendes','George Clooney','Gwyneth Paltrow','Halle Berry',
          'Harrison Ford','Hugh Grant','Jennifer Aniston']

checklist=[]
starttiem,finishtime=0,0
temparray=[]
finalarray=[]
arraylength=0
total_time=''

for i in range(21):
    checklist.append(0)

#인물,시각정보가 기록된 csv파일 로드
file=open('data.csv','r',encoding='utf-8')
readfile=csv.reader(file)
for row in readfile:
    cnt=0
    tempindex=int(row[0])
    checklist[tempindex]=checklist[tempindex]+1
    temparray.append([int(row[0]),row[1]])
    arraylength=arraylength+1

checklist=np.asarray(checklist,dtype=np.int16)

#영상에서 frame으로 읽은 후 가장 많이 도출된 인물로 해당 인물을 지정
person=np.argmax(checklist)

##########################################################
#query문을 전송하기 위한 시각의 자료형 재설정
for i in range(arraylength):
    templist=temparray[i]
    if templist[0]==person:
        finalarray.append(templist)
    
finishindex=len(finalarray)-1
stempstr=finalarray[0][1].split('.')[0]
ftempstr=finalarray[finishindex][1].split('.')[0]
#print(stempstr)

stempdate=stempstr.split(' ')[0]
ftempdate=ftempstr.split(' ')[0]
#print(stempdate)
stemptime_str=stempstr.split(' ')[1]
ftemptime_str=ftempstr.split(' ')[1]
#print(stemptime_str)
startdate=''
finishdate=''
starttime=''
finishtime=''
for i in range(3):
    temp=stempdate.split('-')[i]
    temp2=ftempdate.split('-')[i]
    temp3=stemptime_str.split(':')[i]
    temp4=ftemptime_str.split(':')[i]
    temptime=int(temp4)-int(temp3)
#    print(temptime)
    if 0 < temptime < 9:
        temptime='0'+str(temptime)
    elif temptime==0:
        temptime='00'
    elif temptime < 0:
        temptime=int(temp4)+60-int(temp3)
        temptime=str(temptime)
    startdate=startdate+temp
    finishdate=finishdate+temp2
    starttime=starttime+temp3
    finishtime=finishtime+temp4
    total_time=total_time+str(temptime)+':'
total_time=total_time.rstrip(':')
startdate=int(startdate)
finishdate=int(finishdate)
starttime=int(starttime)
finishtime=int(finishtime)
#print(total_time)
personname=namelist[person]
print(personname)
print(type(personname))
#############################################################
#query문 전송
query="insert into record values (%s,%s,%s,%s,%s,%s)"
curs.execute(query,(personname,startdate,stemptime_str,finishdate,
                    ftemptime_str,total_time))
conn.commit()
print("commit success")
file.close()
#
#
#
#
















