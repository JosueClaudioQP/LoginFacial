#Librerias
import cv2
import face_recognition as fr
import numpy as np
import mediapipe as mp
import os
from tkinter import *
from PIL import Image, ImageTk
import imutils
import math

from PIL.ImageChops import offset

def Profile():
    global step, conteo, UserName, OutFolderPathUser

    #Reset Variables
    step = 0
    conteo = 0

    # Window
    pantalla4 = Toplevel(pantalla)
    pantalla4.title("Profile")
    pantalla4.geometry("1280x720")

    # Fondo
    bc = Label(pantalla4, image=imagenbc, text="Inicio")
    bc.place(x=0, y=0, relwidth=1, relheight=1)

    #File
    UserFile = open(F"{OutFolderPathUser}/{UserName}.txt", "r")
    InfoUser = UserFile.read().split(",")
    Name = InfoUser[0]
    User = InfoUser[1]
    Pass = InfoUser[2]

    #Check User
    if User in clases:
        #Interfaz
        texto1 = Label(pantalla4, text= f"BIENVENIDO {Name}")
        texto1.place(x=580, y=50)

        #Label Img
        lblimage = Label(pantalla4)
        lblimage.place(x=490, y=80)

        #Imagen
        ImgUser = cv2.imread(f"{OutFolderPathFace}/{User}.png")
        ImgUser = cv2.cvtColor(ImgUser, cv2.COLOR_RGB2BGR)
        ImgUser = Image.fromarray(ImgUser)

        IMG = ImageTk.PhotoImage(image=ImgUser)

        lblimage.configure(image=IMG)
        lblimage.image = IMG

#Code Faces Function
def Code_Face(images):
    #List
    listacod = []

    #Iterar por rostro
    for img in images:
        #Color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #Img Code
        cod = fr.face_encodings(img[0])
        #Save List
        listacod.append(cod)

    return listacod

#Close Window
def Close_Window():
    global step, conteo
    #Reset
    conteo = 0
    step = 0
    pantalla2.destroy()

#Close Window
def Close_Window2():
    global step, conteo
    #Reset
    conteo = 0
    step = 0
    pantalla3.destroy()

#Sing Up Biometric Function
def Sign_Biometric():
    global LogUser, LogPass, OutFolderPathFace, cap, lblVideo, pantalla3, FaceCode, clases, images, pantalla2, step, parpadeo, conteo, UserName

    # Check Cap
    if cap is not None:
        ret, frame = cap.read()
        frameSave = frame.copy()
        # Resize
        frame = imutils.resize(frame, width=1280)

        # Frame RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Frame Show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret == True:
            # Inference Face Mesh
            res = FaceMesh.process(frameRGB)

            # Result List
            px = []
            py = []
            lista = []
            if res.multi_face_landmarks:
                # Extract FaceMesh
                for rostros in res.multi_face_landmarks:
                    # Draw
                    mpDraw.draw_landmarks(frame, rostros, FacemeshObject.FACEMESH_CONTOURS, ConfigDraw, ConfigDraw)

                    # Extract Keypoint
                    for id, puntos in enumerate(rostros.landmark):
                        # Info img
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        # 468 Keypoints
                        if len(lista) == 468:
                            # Ojo derecho
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            longitud1 = math.hypot((x2 - x1) + 6, (y2 - y1) + 6)

                            # Ojo izquierdo
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[385][1:]
                            longitud2 = math.hypot((x4 - x3) + 6, (y4 - y3) + 6)

                            # Parietal derecho
                            x5, y5 = lista[139][1:]
                            # Parietal Izquierdo
                            x6, y6 = lista[368][1:]

                            # Ceja derecha
                            x7, y7 = lista[70][1:]
                            x8, y8 = lista[300][1:]

                            # Face detection
                            faces = detector.process(frameRGB)

                            if faces.detections is not None:
                                for face in faces.detections:
                                    # Bbox: "ID, BOX, SCORE"
                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    # Threshold
                                    if score > confThreshold:
                                        # Pixels
                                        xi, yi, anc, alt = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, anc, alt = int(xi * an), int(yi * al), int(anc * an), int(alt * al)

                                        # Offset X
                                        offsetan = (offsetx / 100) * anc
                                        xi = int(xi - int(offsetan / 2))
                                        anc = int(anc + offsetan)
                                        xf = xi + anc

                                        # Offset Y
                                        offsetal = (offsety / 100) * alt
                                        yi = int(yi - offsetal)
                                        alt = int(alt + offsetal)
                                        yf = yi + alt

                                        # Error
                                        if xi < 0:  xi = 0
                                        if yi < 0: yi = 0
                                        if anc < 0: anc = 0
                                        if alt < 0: alt = 0

                                        # Steps
                                        if step == 0:
                                            # Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (255, 0, 255), 2)

                                            # Img Step0
                                            als0, ans0, c = img_step0.shape
                                            frame[50:50 + als0, 50:50 + ans0] = img_step0

                                            # Img Step1
                                            als1, ans1, c = img_step1.shape
                                            frame[50:50 + als1, 1030:1030 + ans1] = img_step1

                                            # Img Step2
                                            als2, ans2, c = img_step2.shape
                                            frame[270:270 + als2, 1030:1030 + ans2] = img_step2

                                            # Face Center
                                            if x7 > x5 and x8 < x6:
                                                # Img Check
                                                alch, anch, c = img_check.shape
                                                frame[165:165 + alch, 1105:1105 + anch] = img_check

                                                # Cont Parpadeo
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                                                    conteo = conteo + 1
                                                    parpadeo = True

                                                elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:
                                                    parpadeo = False

                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070, 375),
                                                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                                                # Cont
                                                if conteo >= 3:
                                                    # IMG Check
                                                    alch, anch, c = img_check.shape
                                                    frame[385:385 + alch, 1105:1105 + anch] = img_check

                                                    # Open Eyes
                                                    if longitud1 > 15 and longitud2 > 15:
                                                        # Step 1
                                                        step = 1
                                            else:
                                                conteo = 0

                                        if step == 1:
                                            #Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (0, 255, 0), 2)
                                            # IMG Check Liveness
                                            alli, anli, c = img_liche.shape
                                            frame[50:50 + alli, 50:50 + anli] = img_liche

                                            #Find Faces
                                            facess = fr.face_locations(frameRGB)
                                            facescod = fr.face_encodings(frameRGB, facess)

                                            #Iteramos
                                            for facecod, faceloc in zip(facescod, facess):

                                                #Mathching
                                                Match = fr.compare_faces(FaceCode, facecod)

                                                #Sim
                                                simi = fr.face_distance(FaceCode, facecod)

                                                #Min
                                                min = np.argmin(simi)

                                                if Match[min]:
                                                    #UserName
                                                    UserName = clases[min].upper()

                                                    Profile()

                                #Close
                                close = pantalla3.protocol("WM_DELETE_WINDOW", Close_Window2)

                            # Circle
                            cv2.circle(frame, (x7, y7), 2, (255, 0, 0), cv2.FILLED)
                            cv2.circle(frame, (x8, y8), 2, (255, 0, 0), cv2.FILLED)

        # Conv Video
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        # Show video
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, Sign_Biometric)

    else:
        cap.release()

#Log Biometric Function
def Log_Biometric():
    global pantalla2, conteo, parpadeo, img_info, step, cap, lblVideo, RegUser

    #Check Cap
    if cap is not None:
        ret, frame = cap.read()
        frameSave = frame.copy()
        #Resize
        frame = imutils.resize(frame, width=1280)

        #Frame RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


        #Frame Show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret == True:
            #Inference Face Mesh
            res = FaceMesh.process(frameRGB)

            #Result List
            px = []
            py = []
            lista = []
            if res.multi_face_landmarks:
                #Extract FaceMesh
                for rostros in res.multi_face_landmarks:
                    #Draw
                    mpDraw.draw_landmarks(frame, rostros, FacemeshObject. FACEMESH_CONTOURS, ConfigDraw, ConfigDraw)

                    #Extract Keypoint
                    for id, puntos in enumerate(rostros.landmark):
                        #Info img
                        al, an, c = frame.shape
                        x, y = int(puntos.x * an), int(puntos.y * al)
                        px.append(x)
                        py.append(y)
                        lista.append([id, x, y])

                        #468 Keypoints
                        if len(lista) == 468:
                            #Ojo derecho
                            x1, y1 = lista[145][1:]
                            x2, y2 = lista[159][1:]
                            longitud1 = math.hypot((x2 - x1)+6, (y2 - y1)+6)

                            #Ojo izquierdo
                            x3, y3 = lista[374][1:]
                            x4, y4 = lista[385][1:]
                            longitud2 = math.hypot((x4 - x3)+6, (y4 - y3)+6)

                            #Parietal derecho
                            x5, y5 = lista[139][1:]
                            #Parietal Izquierdo
                            x6, y6 = lista[368][1:]

                            #Ceja derecha
                            x7, y7 = lista[70][1:]
                            x8, y8 = lista[300][1:]

                            #Face detection
                            faces = detector.process(frameRGB)

                            if faces.detections is not None:
                                for face in faces.detections:
                                    #Bbox: "ID, BOX, SCORE"
                                    score = face.score
                                    score = score[0]
                                    bbox = face.location_data.relative_bounding_box

                                    #Threshold
                                    if score > confThreshold:
                                        #Pixels
                                        xi, yi, anc, alt = bbox.xmin, bbox.ymin, bbox.width, bbox.height
                                        xi, yi, anc, alt = int(xi*an), int(yi*al), int(anc*an), int(alt*al)

                                        #Offset X
                                        offsetan = (offsetx / 100) * anc
                                        xi = int(xi - int(offsetan /2))
                                        anc = int(anc + offsetan)
                                        xf = xi + anc

                                        #Offset Y
                                        offsetal = (offsety / 100) * alt
                                        yi = int(yi - offsetal)
                                        alt = int(alt + offsetal)
                                        yf = yi + alt

                                        #Error
                                        if xi < 0:  xi = 0
                                        if yi < 0: yi = 0
                                        if anc < 0: anc = 0
                                        if alt < 0: alt = 0

                                        #Steps
                                        if step == 0:
                                            #Draw
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (255, 0, 255), 2)

                                            #Img Step0
                                            als0, ans0, c = img_step0.shape
                                            frame[50:50 + als0, 50:50 + ans0] = img_step0

                                            #Img Step1
                                            als1, ans1, c = img_step1.shape
                                            frame[50:50 + als1, 1030:1030 + ans1] = img_step1

                                            #Img Step2
                                            als2, ans2, c = img_step2.shape
                                            frame[270:270 + als2, 1030:1030 + ans2] = img_step2

                                            #Face Center
                                            if x7 > x5 and x8 < x6:
                                                #Img Check
                                                alch, anch, c = img_check.shape
                                                frame[165:165 + alch, 1105:1105 + anch] = img_check

                                                #Cont Parpadeo
                                                if longitud1 <= 10 and longitud2 <= 10 and parpadeo == False:
                                                    conteo = conteo + 1
                                                    parpadeo = True

                                                elif longitud1 > 10 and longitud2 > 10 and parpadeo == True:
                                                    parpadeo = False

                                                cv2.putText(frame, f'Parpadeos: {int(conteo)}', (1070,375), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                                                #Cont
                                                if conteo >= 3:
                                                    #IMG Check
                                                    alch, anch, c = img_check.shape
                                                    frame[385:385 + alch, 1105:1105 + anch] = img_check

                                                    #Open Eyes
                                                    if longitud1 > 15 and longitud2 > 15:
                                                        #Cut
                                                        cut = frameSave[yi:yf, xi:xf]

                                                        #Save Face
                                                        cv2.imwrite(f"{OutFolderPathFace}/{RegUser}.png", cut)

                                                        #Step 1
                                                        step = 1
                                            else:
                                                conteo = 0

                                        if step == 1:
                                            cv2.rectangle(frame, (xi, yi, anc, alt), (0, 255, 0), 2)
                                            #IMG Check Liveness
                                            alli, anli, c = img_liche.shape
                                            frame[50:50 + alli, 50:50 + anli] = img_liche

                                close = pantalla2.protocol("WM_DELETE_WINDOW", Close_Window)

                            #Circle
                            cv2.circle(frame, (x7, y7), 2, (255, 0, 0), cv2.FILLED)
                            cv2.circle(frame, (x8, y8), 2, (255, 0, 0), cv2.FILLED)


        #Conv Video
        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        #Show video
        lblVideo.configure(image=img)
        lblVideo.image = img
        lblVideo.after(10, Log_Biometric)

    else:
        cap.release()

#Funcion Log
def Sign():
    global LogUser, LogPass, OutFolderPathFace, cap, lblVideo, pantalla3, FaceCode, clases, images
    #Extract name, user, pass
    LogUser, LogPass = InputUserLog.get(), InputPassLog.get()

    #DB Faces
    images = []
    clases = []
    lista = os.listdir(OutFolderPathFace)

    #Read Face Images
    for lis in lista:
        #Read Img
        imgdb = cv2.imread(f"{OutFolderPathFace}/{lis}")
        #Save Img DB
        images.append(imgdb)
        #Name Img
        clases.append(os.path.splitext(lis)[0])

    #FaceCode
    FaceCode = Code_Face(images)

    #Window
    pantalla3 = Toplevel(pantalla)
    pantalla3.title("Biometric Sign Up")
    pantalla3.geometry("1280x720")

    # Label video
    lblVideo = Label(pantalla3)
    lblVideo.place(x=0, y=0)

    # Video Capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)
    Sign_Biometric()

#Funcion Log
def Log():
    global RegName, RegUser, RegPass, InputNameReg, InputUserReg, InputPassReg, cap, lblVideo, pantalla2
    #Extract Name - User - Pass
    RegName, RegUser, RegPass = InputNameReg.get(), InputUserReg.get(), InputPassReg.get()

    #Incompleted form
    if len(RegName) == 0 or len(RegUser) == 0 or len(RegPass) == 0:
        #Print error
        print("Formulario Incompleto")
    #Completed form
    else:
        #Check Users
        UserList = os.listdir(PathUserCheck)

        #Name Users
        UserName = []

        #Verificar Check UserList
        for lis in UserList:
            #Extract User
            User = lis
            User = User.split('.')
            #SaveUser
            UserName.append(User[0])

        #Check User
        if RegUser in UserName:
            print("Usuario Registrado Anteriormente")

        else:
            #Save info
            info.append(RegName)
            info.append(RegUser)
            info.append(RegPass)

            #Export file
            f = open(f"{OutFolderPathUser}/{RegUser}.txt", "w")
            f.write(RegName + ",")
            f.write(RegUser + ",")
            f.write(RegPass)
            f.close()

            #Clean
            InputNameReg.delete(0, END)
            InputUserReg.delete(0, END)
            InputPassReg.delete(0, END)

            #New Screen
            pantalla2 = Toplevel(pantalla)
            pantalla2.title("Login Biometric")
            pantalla2.geometry("1280x720")

            #Label video
            lblVideo = Label(pantalla2)
            lblVideo.place(x=0, y=0)

            #Video Capture
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(3, 1280)
            cap.set(4, 720)
            Log_Biometric()

#Patch
OutFolderPathUser = 'C:/Users/JOSUE/Desktop/DataBase/Users'
PathUserCheck = 'C:/Users/JOSUE/Desktop/DataBase/Users/'
OutFolderPathFace = 'C:/Users/JOSUE/Desktop/DataBase/Faces'

#Read img
img_info = cv2.imread("C:/Users/JOSUE/Desktop/SetUp/Info.png")
img_check = cv2.imread("C:/Users/JOSUE/Desktop/SetUp/Check.png")
img_step0 = cv2.imread("C:/Users/JOSUE/Desktop/SetUp/Step0.png")
img_step1 = cv2.imread("C:/Users/JOSUE/Desktop/SetUp/Step1.png")
img_step2 = cv2.imread("C:/Users/JOSUE/Desktop/SetUp/Step2.png")
img_liche = cv2.imread("C:/Users/JOSUE/Desktop/SetUp/LivenessCheck.png")

#Variables
parpadeo = False
conteo = 0
muestra = 0
step = 0

#Offset
offsety = 40
offsetx = 20

#Threshold
confThreshold = 0.5

#Tool Draw
mpDraw = mp.solutions.drawing_utils
ConfigDraw = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

#Object Face Mesh
FacemeshObject = mp.solutions.face_mesh
FaceMesh = FacemeshObject.FaceMesh(max_num_faces=1)

#Object Face Detect
FaceObject = mp.solutions.face_detection
detector = FaceObject.FaceDetection(min_detection_confidence=0.5, model_selection=1)

#InfoList
info = []

#Ventana Principal
pantalla = Tk()
pantalla.title("Face Recognition System")
pantalla.geometry("1280x720")

#Fondo
imageF = PhotoImage(file="C:/Users/JOSUE/Desktop/SetUp/Inicio.png")
background = Label(image = imageF, text = "Inicio")
background.place(x = 0, y = 0, relwidth = 1, relheight = 1)

#Profile
imagenbc = PhotoImage(file="C:/Users/JOSUE/Desktop/SetUp/Back2.png")

#Input Text
#Nombre
InputNameReg = Entry(pantalla)
InputNameReg.place(x=110, y=320)

#User
InputUserReg = Entry(pantalla)
InputUserReg.place(x=110, y=430)

#Contraseña
InputPassReg = Entry(pantalla)
InputPassReg.place(x=110, y=540)

#Input Text SingUp
#User
InputUserLog = Entry(pantalla)
InputUserLog.place(x=750, y=380)

#Contraseña
InputPassLog = Entry(pantalla)
InputPassLog.place(x=750, y=500)

#Button
#Log
imageBR = PhotoImage(file="C:/Users/JOSUE/Desktop/SetUp/BtLogin.png")
BtReg = Button(pantalla, text="Registro", image=imageBR,height="40", width="200", command=Log)
BtReg.place(x=300, y=580)

#Sig
imageBL = PhotoImage(file="C:/Users/JOSUE/Desktop/SetUp/BtSign.png")
BtSing = Button(pantalla, text="Registro", image=imageBR,height="40", width="200", command=Sign)
BtSing.place(x=850, y=580)

pantalla.mainloop()

