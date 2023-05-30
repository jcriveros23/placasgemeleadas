#----------------------------------------MAQUINA DE ESTADOS FINITOS--------------------------------------#

#--------------------------LIBRERIAS------------------------------------#
import cv2
import enum
import time, os
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from collections import defaultdict, Counter
import statistics
import threading
import queue
#-------------------------------------------------------------------------#
#VARIABLES GLOBALES#

#INICIALIZACION DE LAS MAQUINAS#
#--------------------------VEHICULOS-------------------------------------#
weightsPath_vehicles = os.path.sep.join(["redcarros_best.weights"])
configPath_vehicles = os.path.sep.join(["redcarros.cfg"])
net_vehicles = cv2.dnn.readNet(configPath_vehicles, weightsPath_vehicles)
net_vehicles.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_vehicles.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
model_vehicles = cv2.dnn_DetectionModel(net_vehicles)
model_vehicles.setInputParams(size=(928, 544), scale=1/255, swapRB=True)
CONFIDENCE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
class_names = []
with open("carros.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]
#-------------------------------------------------------------------------#
#------------------------------PLACAS-------------------------------------#
weightsPath_plate = os.path.sep.join(["placas_con_sus_carros_best.weights"])
configPath_plate = os.path.sep.join(["placas_con_sus_carros.cfg"])
net_plate = cv2.dnn.readNet(configPath_plate, weightsPath_plate)
net_plate.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_plate.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
model_plate = cv2.dnn_DetectionModel(net_plate)
model_plate.setInputParams(size=(448, 256), scale=1/255, swapRB=True)
class_names_chars = []
#-------------------------------------------------------------------------#
#------------------------------CARACTERES---------------------------------#
weightsPath_char = os.path.sep.join(["placas_cortadas_best.weights"])
configPath_char = os.path.sep.join(["placas_cortadas.cfg"])
net_char = cv2.dnn.readNet(configPath_char, weightsPath_char)
net_char.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net_char.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
model_char = cv2.dnn_DetectionModel(net_char)
model_char.setInputParams(size=(256, 128), scale=1/255, swapRB=True)
tempsito={}
with open("placas_cortadas.names", "r") as f:
    class_names_chars = [cname.strip() for cname in f.readlines()] 
#-------------------------------------------------------------------------#
#------------------------------COLOR--------------------------------------#
img_size = (64, 64)
model = load_model('DNN.h5')
color_names = ["Blanco", "Gris", "Negro", "Rojo", "Amarillo", "Azul", "Verde", "Marron"]
#-------------------------------------------------------------------------#
#------------------------------ESTADOS------------------------------------#
class State(enum.Enum):
    STATE_1 = 1
    STATE_2 = 2
    STATE_3 = 3
    STATE_4 = 4
#-------------------------------------------------------------------------#END INIT

#DEFINICION DE LAS FUNCIONES#
#--------------------------VEHICULOS-------------------------------------#
def DETECTAR_VEHICULOS(frame):
    classes, scores, boxes = model_vehicles.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    img_temp=frame.copy()
    numero_vehiculos=len(boxes)
    cropped_images = []
    labels = [] 
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s" % (class_names[classid])
        cv2.rectangle(frame, box, color, 2)
        #cv2.putText(frame, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        # Recorta la detección y procesa con la segunda red neuronal
        x, y, w, h = box
        cropped_image = img_temp[y:y+h, x:x+w]
        cropped_images.append(cropped_image)
        labels.append(label)
    
    return cropped_images, numero_vehiculos, labels,boxes
#------------------------------
# \-------------------------------------------#
#------------------------------PLACAS-------------------------------------#
def DETECTAR_PLACAS(cropped_images,boxes,labels):
    plate_coords=[]
    placas = []
    Nfails=0
    for i,(car_in_frame, box) in enumerate(zip(cropped_images.copy(),boxes)):
        classes_plate, scores_plate, boxes_plate = model_plate.detect(car_in_frame, 0.7, 0.6)
        numero_placas = len(boxes_plate)
        if  len(boxes_plate)==0:
            print("placa no detectada")
            del cropped_images[i-Nfails]
            boxes=np.delete(boxes,i-Nfails)
            del labels[i-Nfails]
            Nfails+=1
        elif numero_placas>1:
            print("Más de una placa detectada. Se seleccionará la placa con el mayor score.")
            # Encuentra el índice del score máximo
            max_score_idx = np.argmax(scores_plate)
            # Selecciona solo la caja con el score máximo
            boxes_plate = [boxes_plate[max_score_idx]]
            scores_plate = [scores_plate[max_score_idx]]
            
        if len(boxes_plate)>1:
            print("ERRROOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOR")
        x,y,w,h=box
        for (score_plate, box_plate) in zip(scores_plate, boxes_plate):
            color_plate = (255,0,0)
            label_plate = "Placa: %f" % score_plate
            x_plate, y_plate, w_plate, h_plate = box_plate
            plate_coord = [(x + x_plate, y + y_plate), (x + x_plate + w_plate, y + y_plate + h_plate), color_plate, 2]
            plate_coords.append(plate_coord)
            #cv2.putText(frame, label_plate, (x + x_plate, y + y_plate - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color_plate, 2)

            #DETECCION CARACTERES
            #para pasar a red de caracteres
            placa = car_in_frame[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
            placas.append(placa)
            
    return placas, numero_placas, plate_coords
#-------------------------------------------------------------------------#
#------------------------------CARACTERES---------------------------------#
def DETECTAR_CARACTERES(placas):
    license_plate_list = []
    numero_caracteres_list=[]
    
    for(plate_in_car) in placas:

        classes_char, scores_char, boxes_char = model_char.detect(plate_in_car, 0.55, 0.30)
        cont=0
        
        numero_caracteres = len(boxes_char)
        numero_caracteres_list.append(numero_caracteres)
        
        tempsito.clear()
        char_list = []
        labels_char=[]
        
        for (classid_char, score_char, boxes_char) in zip(classes_char, scores_char, boxes_char):
            color_char = COLORS[int(classid_char) % len(COLORS)]
            label_char = "%s" % (class_names_chars[classid_char])
            x_char, y_char, w_char, h_char = boxes_char 
            #cv2.rectangle(frame, (x + x_plate + x_char, y + y_plate +  y_char), (x + x_plate + x_char + w_char, y + y_plate + y_char + h_char), color_char, 2)
            #cv2.putText(frame, label_char, (x + x_plate + x_char, y + y_plate  + y_char), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_char, 2)
            char_list.append((x_char, class_names_chars[classid_char]))
            labels_char.append(label_char)

           
        # Ordenar la lista de caracteres según la posición en x
        sorted_char_list = sorted(char_list, key=lambda x: x[0])
        # Guardar los caracteres ordenados en un diccionario
        for i, (x_char, label_char) in enumerate(sorted_char_list):
            tempsito[i] = label_char
        char_values = tempsito.values()
        # Concatenar los caracteres en un solo string
        license_plate = "".join(char_values)
        license_plate_list.append(license_plate)
    
    return license_plate_list, numero_caracteres_list
#-------------------------------------------------------------------------#
#-------------------------------COLOR-------------------------------------#
def DETECTAR_COLOR(cropped_images):
    #cropped_image = cropped_image.copy()
    predicted_color_names=[]
    for(cropped_image)in cropped_images:
        img = cv2.resize(cropped_image, img_size)
        img = img / 255.0  # Normalización
        img = np.expand_dims(img, axis=0)  # Agregar una dimensión adicional para que sea compatible con el modelo

    # Utilizar el modelo para predecir el color de los datos de prueba
        predictions = model.predict(img)
        predicted_labels = np.argmax(predictions, axis=1)
        predicted_color_name = color_names[predicted_labels[0]]
        predicted_color_names.append(predicted_color_name)
    return predicted_color_names

def update_gui(image_label2,image2, root):
    #image_label.config(image=image)
    #image_label.image = image
    image_label2.config(image=image2)
    image_label2.image = image2 
    
#-------------------------------------------------------------------------#
#--------------------------------MAIN-------------------------------------#
def main(video_path,image_queue,ui):
    #video_path = "video_prueba4.avi"
    cont=0
    cap = cv2.VideoCapture(video_path)
    frames = []
    detections=[]
    all_cropped_images=[]
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    current_state = State.STATE_1

    for frame_index, frame in enumerate(frames):
        cont+=1
        fluxo = time.time()
        
        if ui.stop_processing:
            return detections, all_cropped_images
        
        if current_state == State.STATE_1:
            cropped_images, numero_vehiculos, labels,boxes=DETECTAR_VEHICULOS(frame)
            
            if numero_vehiculos > 0:
                current_state = State.STATE_2
            else:
                current_state = State.STATE_1
            
        if current_state == State.STATE_2:
            placas, numero_placas,plate_coords = DETECTAR_PLACAS(cropped_images,boxes,labels)
            
            if numero_placas > 0:
                current_state = State.STATE_3
            else: 
                current_state = State.STATE_1
            #print(current_state)
        if current_state == State.STATE_3:
            license_plate_list, numero_caracteres_list = DETECTAR_CARACTERES(placas)
            Nfails=0
            for i, (numero_caracteres) in enumerate( numero_caracteres_list.copy()):
            
                if numero_caracteres != 6:
                    
                    del license_plate_list[i-Nfails]
                    del numero_caracteres_list[i-Nfails]
                    del cropped_images[i-Nfails]
                    del labels[i-Nfails]
                    boxes=np.delete(boxes,i-Nfails)
                    Nfails+=1
            #print(numero_caracteres_list)
            if(numero_caracteres_list):
                current_state = State.STATE_4
            else:
                current_state = State.STATE_1 
            #print(current_state) 
        if current_state == State.STATE_4:
            colors = DETECTAR_COLOR(cropped_images)
            for indice, (cropped_image, label, caracteres, color) in enumerate(zip(cropped_images, labels, license_plate_list, colors)):
                print("El carro es de tipo: " + label + ", con placa: "+ caracteres + ", de color: " + color)
                all_cropped_images.append(cropped_image)
                detections.append([caracteres,label,color,len(all_cropped_images)-1,frame_index])
                
                current_state = State.STATE_1
                image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)  # Convertir la imagen de BGR a RGB
                image = cv2.resize(image, (448, 256))
                image = Image.fromarray(image)  # Convertir la imagen de OpenCV a una imagen de PIL
                image = ImageTk.PhotoImage(image)  # Convertir la imagen de PIL a PhotoImage 
                image_queue.put(("cropped", image)) # Agregar la imagen a la cola en lugar de actualizar la GUI directamente
            
        """ update_thread = threading.Thread(target=update_gui, args=(image_label2,image2,root))
        update_thread.start() """
        """ if root:
            root.update_idletasks() """ 
        image2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir la imagen de BGR a RGB
        image2 = cv2.resize(image2, (928, 544))
        image2 = Image.fromarray(image2)  # Convertir la imagen de OpenCV a una imagen de PIL
        image2 = ImageTk.PhotoImage(image=image2)  # Convertir la imagen de PIL a PhotoImage
        image_queue.put(("main", image2)) # Agregar la imagen a la cola en lugar de actualizar la GUI directamente
        #image_label2.config(image=image2)
        #image_label2.image = image2 
        #print("FPS: ", 1.0 / (time.time() - fluxo))
        fps=1.0 / (time.time() - fluxo)
        ui.fps_label.config(text=f'FPS: {fps:.2f}')
        
        
    #print(detections)
    return detections, all_cropped_images
    #cv2.destroyAllWindows()
#-------------------------------------------------------------------------#