import cv2
import os
from face_detection import face
from keras.models import load_model
import numpy as np
#from embedding import emb
from retreive_pymongo_data import database


label=None
a={0:0,1:0,2:0,3:0}
people={0:"vasan",1:"vishal",2:"yash",3:"sathyaveer"}
abhi=None
data=database()
fd=face()

print('Attendance till now is ')
data.view()

model=load_model('siamese_net.MODEL')

PATH = os.getcwd()
# Define data path
data_path = PATH + "/data"
people=os.listdir('data')

def test():
    pre = []
    for x in people:
        print("Loading test-set for - {}".format(x[:-1]))
        img=cv2.imread('data'+'/'+x+'/'+'5.jpg',3)
        img=cv2.resize(img,(160,160))
        img=img.astype('float')/255.0
        #img=np.expand_dims(img,axis=0)
        pre.append(img)
    pre = np.array(pre) 
       
    print(pre.shape[:])
    return pre
    #test_run=e.calculate(test_run)
    #test_run=np.expand_dims(test_run,axis=0)
    #test_run=model.predict(test_run)[0]


cap=cv2.VideoCapture(0)
ret=True
pre = test()
while ret:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    detected,x,y,w,h=fd.detectFace(frame)

    if(detected is not None):
        f=detected
        detected=cv2.resize(detected,(160,160))
        detected=detected.astype('float')/255.0
        detected=np.expand_dims(detected,axis=0)
        detected=np.array(detected)
        detected=np.expand_dims(detected,axis=0)
        print(detected.shape[:])
        #feed=e.calculate(detected)
        #feed=np.expand_dims(feed,axis=0)
        for x in range(len(pre)):
            prediction=model.predict([pre[x],detected[:]])
            if(prediction==1):
            	result = x
            	#break       
        for i in people:
            if(result==i):
                label=people[i]
                if(a[i]==0):
                    data.update(label)
                a[i]=1
                abhi=i
        #data.update(label)
        print(a[:]+"\n")
        cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        if(a[abhi]==1):
            cv2.putText(frame,"your attendance is complete",(x,y-30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(252,160,39),3)
        cv2.imshow('onlyFace',f)
    cv2.imshow('frame',frame)
    if(cv2.waitKey(1) & 0XFF==ord('q')):
        break
cap.release()
cv2.destroyAllWindows()
data.export_csv()