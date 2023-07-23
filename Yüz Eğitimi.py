#########################################
# yüz verilerinin eğitimi

import cv2
import os
import numpy as np
from PIL import Image


path = "ogrenci_"
paths = "ogrenci_verileri"


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create() 
# opencv nin yüz tanıyıcı fonksiyonu 2 dizi döndürür 1. si görüntüler ve 2. si görüntülerin id leri


faceSamples=[]
ids = []
def getImagesAndLabels(path_number):
    
        
    imagePaths = [os.path.join(paths+"/"+path_number,f) for f in os.listdir(paths+"/"+path_number)]     
    global faceSamples
    global ids 
    
    for imagePath in imagePaths:
    
        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')
    
        id = int(os.path.split(path_number)[-1].split("_")[1])
        faces = face_cascade.detectMultiScale(img_numpy)
    
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
       
    return faceSamples,ids


countFolder = 1


while os.path.exists(paths+"/"+path + str(countFolder)):
    path_number = path + str(countFolder)
    faceSamples,ids = getImagesAndLabels(path_number)
    recognizer.train(faceSamples, np.array(ids))
    countFolder +=1

    
print ("\n  Eğitim tamamlanıyor lütfen bekleyiniz ...")

    
# Tanıyıcımızı  trainer/trainer.yml formatında kaydettik recognizer.write raspberry pi de kullanılıyor.
recognizer.save('trainer'+"/"+'trainer'+'.yml') # recognizer.save() worked on Mac, but not on Pi
    
# Print the numer of faces trained and end program
print("\n  {0} adet yüz eğitildi. Çıkış yapabilirsiniz".format(len(np.unique(ids))))


