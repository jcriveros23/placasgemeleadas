import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from maquina_estados import main
import os
import sys
import queue
import threading
from PIL import Image, ImageTk
import numpy as np
import cv2
from collections import defaultdict, Counter
import statistics
import time
import pandas as pd

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.title("GUI Example")
        self.master.geometry("1920x1080")
        self.default_image2 = tk.PhotoImage(file="portada.gif")
        self.default_image = ImageTk.PhotoImage(Image.open("javelogo2.png"))
        self.create_widgets()
        self.root = master
        self.stop_processing = False
        self.image_queue = queue.Queue()
        
        self.master.after(100, self.update_image)

    def create_widgets(self):
        # Create menu bar
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)

        # Create file menu
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Reboot", command=self.reboot_gui)
        filemenu.add_command(label="Exit", command=self.master.quit)
        
        menubar.add_cascade(label="File", menu=filemenu)

        # Create buttons
        self.video_label = ttk.Label(self.master, text="")
        self.video_label.place(relx=0.58, rely=1, y=-10, anchor='sw')

        self.fps_label = ttk.Label(self.master, text="")
        self.fps_label.place(relx=0,rely=0.8,y=10,anchor='nw')
        
        self.alert_label = ttk.Label(self.master,text="",foreground="red")
        self.alert_label.place(relx=0.7, rely=0.75, anchor='sw')     

        self.upload_video_button = ttk.Button(self.master, text="Upload video", command=self.upload_video)
        self.upload_video_button.place(relx=0, rely=1, x=10, y=-10, anchor='sw')

        self.output_text = tk.Text(self.master, wrap='word', height=10, width=50)
        self.output_text.place(relx=1, rely=0.5, x=-10, y=10, anchor='e')

        self.run_button = ttk.Button(self.master, text="RUN", command=self.run_script, state=tk.DISABLED)
        self.run_button.place(relx=0.15, rely=1, x=10, y=-10, anchor='sw')

        self.show_results_button = ttk.Button(self.master, text="Show Results", command=self.show_results, state=tk.DISABLED)
        self.show_results_button.place(relx=0.3, rely=1, x=10, y=-10, anchor='sw')

        self.stop_button = ttk.Button(self.master, text="STOP", command=self.stop, state=tk.DISABLED)
        self.stop_button.place(relx=1, rely=1, x=-50, y=-10, anchor='se')

        self.image_label = ttk.Label(self.master, image=self.default_image)
        self.image_label.place(relx=1, rely=0, x=-10, y=10, anchor='ne')

        self.image_label2 = ttk.Label(self.master, image=self.default_image2)
        self.image_label2.place(relx=0, rely=0, x=10, y=10, anchor='nw')

        # Agrega el control deslizante en la ventana principal
        self.detection_slider = ttk.Scale(self.master, from_=0, to=100, length=200, orient='horizontal', command=self.update_frame, state=tk.DISABLED)
        self.detection_slider.place(relx=0.5, rely=1, x=-150, y=-10, anchor='sw')

    def do_nothing(self):
        pass
    def stop(self):
        self.stop_processing = True

    def reboot_gui(self):
        # Reset button states
        self.run_button.config(state='normal')
        self.upload_video_button.config(state='normal')
        self.show_results_button.config(state='normal')
        self.detection_slider.config(state='disabled')

        # Clear text and labels
        self.output_text.delete('1.0', tk.END)
        self.video_label.config(text="")
        self.image_label.config(image=self.default_image)   
        self.image_label.image = self.default_image
        self.image_label2.config(image=self.default_image2)
        self.image_label2.image = self.default_image2

        # Clear variables
        self.summary = {}
        self.detections = []
        self.all_cropped_images = []
        self.total_detections = 0
        self.file_path = None

        # Clear image queue
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except queue.Empty:
                break

    
    def load_frames(self):
        cap = cv2.VideoCapture(self.file_path)
        # Obteniendo el número total de frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
        # Obteniendo los FPS (frames por segundo)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Calculando la duración en segundos
        duration = total_frames / fps
        frames = []
        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        
        return frames, fps

    def process_detections(self,detections,all_cropped_images):
        ##Procesamiento de resultados:
        body_types = defaultdict(list)
        colors = defaultdict(list)
        image_indices = defaultdict(list)
        first_seen = {}  # Diccionario para almacenar el primer tiempo de detección para cada vehículo

        # Agrupa las detecciones por matrícula
        for plate, body_type, color, image_index, frame_idx  in detections:
            body_types[plate].append(body_type)
            colors[plate].append(color)
            image_indices[plate].append(image_index)
            if plate not in first_seen:
                first_seen[plate] = (frame_idx) / self.fps  # Agregar el tiempo de la primera detección al diccionario

        # Calcula la moda de las carrocerías y colores por matrícula y almacena los resultados en un nuevo diccionario
        summary = {}
        for plate in body_types.keys():
            body_type_mode = Counter(body_types[plate]).most_common(1)[0][0]
            color_mode = Counter(colors[plate]).most_common(1)[0][0]
            resultcheck,real_body,real_color = self.check_vehicle(plate, body_type_mode, color_mode)
            if(resultcheck == "Posible vehiculo clonado"):
                self.alert_label.config(text="ALERTA: posibles vehiculos clonados \nPor favor revisar resultados", font=("Helvetica", 18),foreground="red")

            if image_indices[plate]:
                integer_indices = [int(index) for index in image_indices[plate]]
                image_index_median = sorted(integer_indices)[len(integer_indices) // 2]
                if 0 <= image_index_median < len(all_cropped_images):
                    summary[plate] = (body_type_mode, color_mode, all_cropped_images[image_index_median], first_seen[plate],resultcheck,real_body,real_color)
                else:
                    # Si el índice está fuera de rango, usa un valor predeterminado (por ejemplo, una imagen en blanco)
                    default_image = Image.new("RGB", (128, 128), color="white")
                    summary[plate] = (body_type_mode, color_mode, default_image, first_seen[plate],resultcheck,real_body,real_color)
            else:
                # Si no hay índices en la lista, usa un valor predeterminado (por ejemplo, una imagen en blanco)
                default_image = Image.new("RGB", (128, 128), color="white")
                summary[plate] = (body_type_mode, color_mode, default_image, first_seen[plate],resultcheck,real_body,real_color)

        unique_detections = list(summary.values())
        #print(unique_detections)
        return summary

    def check_vehicle(self,plate, body_type_mode, color_mode):
        # Leer el archivo CSV
        df = pd.read_csv('datatest_det.csv', sep=';')
        
        # Filtrar el DataFrame para obtener solo las filas con la placa especificada
        vehicle_rows = df[df['Placa'] == plate]
        
        # Si no hay vehículos con esa placa, devolver un string vacío
        if vehicle_rows.empty:
            return "Vehiculo no encontrado","",""
        
        # Obtener los valores de carrocería y color de los vehículos con la placa especificada
        vehicle_body_type = vehicle_rows['Carroceria'].values[0]
        vehicle_color = vehicle_rows['Color'].values[0]
        
        # Si la carrocería y el color coinciden con los modos proporcionados, devolver un string vacío
        if vehicle_body_type != body_type_mode or vehicle_color != color_mode.lower():
            #print(vehicle_body_type,"vs",body_type_mode," or ",vehicle_color,"vs",color_mode," in", plate )
            return "Posible vehiculo clonado", vehicle_body_type, vehicle_color

        
        # Si ninguno de los condicionales anteriores se cumple, devolver un string vacío
        return "","",""
    
    def run_script(self):
        self.detection_slider.config(state=tk.DISABLED)
        self.show_results_button.config(state='disabled')
        self.run_button.config(state='disabled')
        self.upload_video_button.config(state='disabled')
        self.stop_button.config(state='normal')
        # Cambiar el texto del video_label a "Un momento... Se está procesando el video"
        self.video_label.config(text="Un momento... Se está procesando el video")
        self.alert_label.config(text="")
        self.master.update()  # Actualizar la GUI para mostrar el cambio de texto
        self.output_text.delete('1.0', tk.END)  # Clear text widget

        sys.stdout = TextRedirector(self.output_text, "stdout")
        sys.stderr = TextRedirector(self.output_text, "stderr")
        # Crear un hilo para ejecutar el procesamiento de imágenes
        image_processing_thread = threading.Thread(target=self.run_vehicle_detections)
        image_processing_thread.start()
        
    def run_vehicle_detections(self):
        self.detections, self.all_cropped_images = main(self.file_path, self.image_queue,self)

        self.frames,self.fps = self.load_frames()
        self.summary = self.process_detections(self.detections, self.all_cropped_images)
        self.total_detections = len(self.all_cropped_images)
        self.detection_slider.config(to=self.total_detections - 1)

        self.video_label.config(text="Video procesado con éxito")
        self.detection_slider.config(state=tk.NORMAL)
        self.show_results_button.config(state='normal')
        self.run_button.config(state='normal')
        self.upload_video_button.config(state='normal')
        self.stop_processing=False
        self.stop_button.config(state='disabled')

    def update_frame(self, detection_number):
        detection_number = int(float(detection_number))  # Convierte el valor del slider a entero
        # Actualiza las detecciones en función del número de frame proporcionado
        self.show_frame_with_detections(detection_number)

    def show_frame_with_detections(self, detection_number):
        # Muestra el frame con las detecciones dibujadas en la ventana principal
        if 0 <= detection_number < self.total_detections:
            img1 = self.all_cropped_images[detection_number]
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1 = Image.fromarray(img1)
            img1 = img1.resize((448, 256), Image.LANCZOS)
            imgtk1 = ImageTk.PhotoImage(img1)

            self.image_label.config(image=imgtk1)
            self.image_label.image = imgtk1

            plate, body_type, color, image_index, frame_idx=self.detections[detection_number]
            img2 = self.frames[int(frame_idx)]
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = Image.fromarray(img2)    
            img2 = img2.resize((928, 544), Image.LANCZOS)
            imgtk2 = ImageTk.PhotoImage(img2)

            self.image_label2.config(image=imgtk2)
            self.image_label2.image = imgtk2

            self.fps_label.config(text=f"Tiempo: {(frame_idx) / self.fps}")
            
            print("El carro es de tipo: " + body_type + ", con placa: "+ plate + ", de color: " + color)
        else:
            self.image_label.config(image='')
            self.image_label.image = None
            self.image_label2.config(image='')
            self.image_label2.image = None

    def upload_video(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi;*.mp4")])
        if self.file_path:
            video_name = os.path.basename(self.file_path)
            self.video_label.config(text=f"Video cargado: {video_name}")
            print("video uploaded")
            self.run_button.config(state='normal')
        return self.file_path
    
    def show_results(self):
        results_window = tk.Toplevel(self.master)
        results_window.title("Results")
        results_window.geometry("900x800")

        # Create scrollbar and canvas
        scrollbar = ttk.Scrollbar(results_window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas = tk.Canvas(results_window, yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=canvas.yview)

        # Create frame inside the canvas
        results_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=results_frame, anchor='nw')

        # Populate the frame with detection results
        for i, (plate, (body_type, color, image, first_detected,resultcheck,real_body,real_color)) in enumerate(self.summary.items()):
            # Convert the numpy array to a PIL image
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img.thumbnail((350, 350), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(img)

            image_label = ttk.Label(results_frame, image=imgtk)
            image_label.image = imgtk
            image_label.grid(row=i*4, column=0, rowspan=4, padx=5, pady=5)

            plate_label = ttk.Label(results_frame, text=f"Matrícula: {plate}", font=("TkDefaultFont", 16))
            plate_label.grid(row=i*4, column=1, sticky="nw", padx=5, pady=5)

            body_type_label = ttk.Label(results_frame, text=f"Tipo de carrocería: {body_type}", font=("TkDefaultFont", 16))
            body_type_label.grid(row=i*4+1, column=1, sticky="nw", padx=5, pady=5)

            color_label = ttk.Label(results_frame, text=f"Color: {color}", font=("TkDefaultFont", 16))
            color_label.grid(row=i*4+2, column=1, sticky="nw", padx=5, pady=5)

            # Add a label for the first detected time
            first_detected_label = ttk.Label(results_frame, text=f"Deteccion (segundos): {first_detected:.2f}", font=("TkDefaultFont", 16))
            first_detected_label.grid(row=i*4+3, column=1, sticky="nw", padx=5, pady=5)

            #print(i,resultcheck,plate,real_body,real_color)
            
            if resultcheck=="Posible vehiculo clonado":
                alert_result_label = ttk.Label(results_frame, text=f"ALERTA: {resultcheck}\nCarrocería Real: {real_body}\nColor Real: {real_color}", font=("TkDefaultFont", 16),foreground="red")
                alert_result_label.grid(row=i*4, column=2, sticky="nw", padx=5, pady=5)
            if resultcheck=="Vehiculo no encontrado":
                alert_result_label = ttk.Label(results_frame, text=f"NOTA: {resultcheck}", font=("TkDefaultFont", 16),foreground="black")
                alert_result_label.grid(row=i*4, column=2, sticky="nw", padx=5, pady=5)
        # Update canvas and set scroll region
        results_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox('all'))
        if not self.summary:
            no_results_label = ttk.Label(results_frame, text="No se han obtenido resultados o el script aún se está ejecutando.")
            no_results_label.grid(row=0, column=0, padx=5, pady=5, columnspan=4)

    def update_image(self):
        try:
            image_type, image = self.image_queue.get_nowait()  # Obtiene la imagen más reciente y su tipo de la cola
            if image_type == "main":
                self.image_label2.config(image=image)
                self.image_label2.image = image
            elif image_type == "cropped":
                self.image_label.config(image=image)
                self.image_label.image = image
            self.master.after(100, self.update_image)  # Programa la próxima actualización en 500 ms (0.5 segundos)
        except queue.Empty:
            self.master.after(100, self.update_image)  # Programa la próxima actualización si la cola está vacía

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.see("end")
        self.widget.configure(state="disabled")

    def flush(self):
        pass


root = tk.Tk()
app = Application(master=root)
app.mainloop()
