#encoding:utf-8
'''
程序修改思路2018.6.14
问题：程序运行速度过慢，分割不准
解决放案：改成颜色分割，像素降低
拐弯方向的解决放案:固定一个方向转，如果那侧没有盲道，则再转180度
改进：如果一张图里面有两条直线以上则认为有盲道emm。。。
'''
import numpy as np
import ConfigParser
import cv2
import PIL.Image as Image
from sklearn.cluster import KMeans
import math
from time import time
'''上车之后这里修改成视频流'''
#from xunfei_voice.conversation import Conversation as conv
count = 0
#Speak = conv()
cf = ConfigParser.ConfigParser()
cf.read('/home/lixuan/camera.conf')
choose = int(cf.get('db','db_mangdao'))
flag1 = 0
flag2 = 0
count1 = 0
from socket import socket, AF_INET, SOCK_DGRAM
ADDR = 'localhost'
PORT = 20000

class socket_sender(object):
    def __init__(self, addr, port):
        self.sock = socket(AF_INET, SOCK_DGRAM)
        self.addr = addr
        self.port = port

    def send_data(self, data):
        self.sock.sendto(data, (self.addr, self.port))

sender = socket_sender('192.168.20.136',20000)
sender_to_voice = socket_sender('localhost',20000)

def cal_angle(x1,y1,x2,y2):
    if (y2-y1) == 0:
        return 90.0#90度认为平行于x轴
    else:
        angle = abs(math.degrees(math.atan((x2-x1)/(y2-y1))))#这里我是做的对y的夹角
        #print angle
        return angle
cap = cv2.VideoCapture(choose)
#设置腐蚀
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
while (True):
    start = time()
    ret,image = cap.read()
    image = cv2.pyrDown(image)
    #image =  cv2.erode(image,kernel)
    #image = cv2.imread("318.jpg")
    #cv2.imshow("h",image)
    #转换lab空间,并且对图像进行预处理:灰度腐蚀
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    #cv2.imshow("L*a*b*", lab)
    #lab1 = eroded = cv2.erode(lab,kernel)
    #L,A,B = cv2.split(lab1,mv=None)

    #cv2.imwrite("C.jpeg",lab1)  
    # 像素的提取,特征提取  
    img = Image.fromarray(lab)
    data = []
    #img = image.open(f)
    row,col = img.size
    for i in range(row):
        for j in range(col):
            x,y,z = img.getpixel((i,j))
            data.append([x/256.0,y/256.0,z/256.0])
    #imgData,row,col = loadData('C.jpeg')
    imgData = np.mat(data)


    #kmeans聚类
    '''这个地方通过修改该聚类中心cluster来改变环境影响,而且可以更改kmeans的迭代次数这里我选用默认的'''
    label = KMeans(n_clusters=2).fit_predict(imgData)
    label = label.reshape([row,col])
    pic_new = Image.new("L", (row, col))
    for i in range(row):
        for j in range(col):
            pic_new.putpixel((i,j), int(256/(label[i][j]+1)))
    pic_new.save("result-bull-4.jpg", "JPEG")
    #print pic_new

    #hough变换求直线
    img = cv2.imread("result-bull-4.jpg")  
    img = cv2.GaussianBlur(img,(3,3),0)  
    edges = cv2.Canny(img, 50, 150, apertureSize = 3)  
    #第一个参数是一个二值化图像 所以在进行霍夫变换之前  先进行  二值化 或者  Canny  缘检测。第二和第三个值分别代  ρ 和 θ 的精确度。第四个参数是阈值， 只有累加其中的值高于阈值时才被认为是一条直线 
    #lines = cv2.HoughLines(edges,1,np.pi/180,118) #这里对最后一个参数使用了经验型的值
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
    try:
        lines1 = lines[:,0,:]#提取为二维
    #print lines1
        result = img.copy()
        #flag1 = 1
        for x1,y1,x2,y2 in lines1[:]: 
            cv2.line(result,(x1,y1),(x2,y2),(255,0,0),3)  
            angle_temple = cal_angle(x1,y1,x2,y2)
            if angle_temple > 60 and flag2==0:#这个是阈值是否拐弯的
                count += 1
                flag1 = 1
                if count >=2:
                    if flag2 ==0:
                        #Speak.tts_play("拐弯") #认为两条以上的边界是拐弯的地方 其他的可能是条纹
                        sender_to_voice.send_data("拐弯")
                        #ss, addr = s.accept() 
                        #print 'got connected from',addr 
                        #ss.send('turn')
                        sender.send_data('turn')
                        flag2 =1
                        count=0
                        break
            else:
                count1+=1
                if count1 >=8:
                    flag2 = 0
                    count1 = 0
                    break
            '''
            这里面还得该因为还得等车转弯完成防止转过头,这里面的参数可以通过上车调试之后才能确定
            用延时或者再判断一遍较角度来实现
            '''
                
                    #这里要转90
                    #break#防止一直拐弯
            print angle_temple
        '''
        一些改进想法这个地方要做角度的计算来确定是否拐弯
        '''
        if flag1 == 0:
            count = 0
        else:flag1 = 0
        #cv2.imshow('Canny', edges)  
        cv2.imshow('Result', result)
        sender_to_voice.send_data('发现盲道') 
    except:
        print "没有盲道" 
    stop = time()
    print(str(stop-start) + "秒")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()  
cv2.destroyAllWindows() 
