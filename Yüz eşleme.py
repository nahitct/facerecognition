import cv2
import os 
import time

recognizer = cv2.face.LBPHFaceRecognizer_create()


# trained = 1
# while os.path.exists('trainer'+"/"+'trainer'+str(trained)+'.yml'):
#    recognizer.read('trainer'+"/"+'trainer'+str(trained)+'.yml')
#    trained +=1

recognizer.read('trainer'+"/"+'trainer'+'.yml')

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#kimlik sayacı
id = 0

# kimlik numarası ve isim eşleştirmek için isimleri alıyoruz.
names = []
path_number = 1
path = "ogrenci_"
paths = "ogrenci_verileri"


while os.path.exists(paths+"/"+path + str(path_number)):
    imagePaths =  [os.path.join(paths+"/"+str(path_number),f) for f in os.listdir(paths+"/"+path+str(path_number))] 
    name = str (imagePaths[1])
    name =(os.path.split(name)[-1].split(".")[0])
    names.append(name)
    path_number += 1



# Gerçek zamanlı kamera ayarları
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht, görüntü genişliği
cam.set(4, 480) # set video height, görüntü yüksekliği

# yüz tespiti için minimum pencere boyutu 
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()
  

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
 
        if (confidence < 100):
            id = names[id-1]
            confidence = "  {0}%".format(round(confidence))
        else:
            id = "Yabanci"
            confidence = "  {0}%".format(round(confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 
    time.sleep(0.2)
    k = cv2.waitKey(20) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break


print("Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()

