# https://youtu.be/pI0wQbJwIIs



import numpy as np
#from PIL import Image
#from sklearn.preprocessing import LabelEncoder
#from tensorflow.keras.models import load_model


from tensorflow.keras.models import load_model
# importing askopenfile function
# from class filedialog
from tkinter.filedialog import askopenfile
import numpy as np
from ultralytics import YOLO

import tensorflow as tf
import cv2
import keras
from keras.models import load_model
import os



def detect2(frame):
  model = YOLO(r"C:\Users\Marize\yolov8n.pt")
  results = model(frame)  
  c=results[0].boxes.cls.cpu().numpy().astype(int)
  try:
    for I in c:
      if I==0:
        return 1
        break
      else:
        return 0

  except:
      print("not detected")
      return 0 
  
    
def get_frames(file_name):
    images = []

    vidcap = cv2.VideoCapture(file_name)
    
    success,image = vidcap.read()
        
    count = 0
    while success:
        test=detect2(image)
        if test==1:
          RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          res = cv2.resize(RGB_img, (128,128))
          images.append(res)
          count += 1
          if count==16:
            break
        success,image = vidcap.read()
    resul = np.array(images)
    resul = (resul / 255.).astype(np.float16)
    return resul




def final(filename):
    frames=get_frames(filename)
    model=load_model(r"C:\Users\Marize\Desktop\gp final\app\models\3dmod.hdf5")
    model.summary()
    re=np.reshape(frames,(1,16,128,128,3))
    r=model.predict(re)
    r = r.flatten()
    print(r.round(2))
    # extract the predicted class labels
    r = np.where(r > 0.5, 1, 0)
    c=[]
    c.append([frames,r])
    return c


    
def getPrediction(filename):
    classes = ['NonViolence','Violence']
    #le = LabelEncoder()
    #le.fit(classes)
    #le.inverse_transform([2])
    
    #Load model
    #my_model=load_model("model/HAM10000_100epochs.h5")
    
    Video_path = './images/'+filename
    print(Video_path)
    pre=final(Video_path)
    Y = np.array([j[1] for j in pre])
    X = np.array([j[0] for j in pre])
    print(X)
    print(Y)

    if Y==0:
        return "nonviolence"
    else:
        return "violence"



#test_prediction =getPrediction('example.jpg')

