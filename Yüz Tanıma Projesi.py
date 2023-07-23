import cv2
import os
import time


#######################################################################################
# yüz verilerini toplama

# sınıflandırıcı
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# video
cap = cv2.VideoCapture(0)


path = "ogrenci_"
paths = "ogrenci_verileri"
isim = input("Öğrenci ismi(lütfen Türkçe karakter kullanmayınız (ş,ö,ı,ğ,ü,ç)): ")

global  path_number, countFolder
def saveDataFunc():
    global  path_number, countFolder
    countFolder = 1
    while os.path.exists(paths+"/"+path + str(countFolder)):
        countFolder += 1
        
    path_number = path + str(countFolder) 
    os.makedirs(paths+"/"+path_number)
    return path_number
    

saveDataFunc()
countSave = 1



while True:
    
    ret, frame = cap.read()
    
    if ret :
        
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rect = face_cascade.detectMultiScale(frame, 1.3, minNeighbors = 7)
               
        for (x,y,w,h) in face_rect:
               
            cv2.rectangle(frame, (x,y),(x+w, y+h),(255,255,255),10)
              
            print(countSave)
            cv2.imwrite(paths+"/"+path+str(countFolder)+"/"+isim+"."+str(countSave)+".png",frame[y:y+h,x:x+w])
            countSave += 1
            
                
          
    time.sleep(0.2)    
    cv2.imshow("Yuz Kaydi", frame)
            

                 
    if cv2.waitKey(100) & 0xFF == ord("q") :
        break
    elif countSave > 50:
        break
    
cap.release()
cv2.destroyAllWindows()


 
















