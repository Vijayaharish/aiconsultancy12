from flask import Flask
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for, jsonify
from camera import VideoCamera
from camera2 import VideoCamera2
import os
import time
import datetime
from random import randint
import cv2
import PIL.Image
from PIL import Image
import imagehash
from flask import send_file
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import docx2txt
import shutil
import subprocess
import gensim
#word to pdf
import aspose.words as aw

import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from gensim.parsing.porter import PorterStemmer
import re

import pyttsx3
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="",
  charset="utf8",
  database="ai_consultancy"
)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####

@app.route('/',methods=['POST','GET'])
def index():
    act=""
    msg=""

    #now1 = datetime.datetime.now()
    #rtime=now1.strftime("%H:%M")
    #print(rtime)

    return render_template('web/index.html',msg=msg,act=act)

@app.route('/login_user', methods=['GET', 'POST'])
def login_user():
    msg=""
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM ac_user WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            ff=open("uname.txt","w")
            ff.write(uname)
            ff.close()

            ff=open("emotion.txt","w")
            ff.write("")
            ff.close()

            cursor.execute("update ac_data set status=0,response2='',qno=0")
            mydb.commit()

            cursor.execute("delete from ac_temp where uname=%s",(uname,))
            mydb.commit()
    
            return redirect(url_for('user_test'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('web/login_user.html',msg=msg)

@app.route('/login_doc', methods=['GET', 'POST'])
def login_doc():
    msg=""
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM ac_doctor WHERE username = %s AND password = %s AND approved_status=1', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            ff=open("doc.txt","w")
            ff.write(uname)
            ff.close()

            fnn=uname+".txt"
            #ff=open("static/"+fnn,"w")
            #ff.write("")
            #ff.close()
            
            return redirect(url_for('doc_home'))
        else:
            msg = 'Incorrect username/password! or Not Approved'
    return render_template('web/login_doc.html',msg=msg)

@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""
    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM ac_admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('web/login.html',msg=msg)



@app.route('/register',methods=['POST','GET'])
def register():
    msg=""
    act=""
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        
        uname=request.form['uname']
        pass1=request.form['pass']
      
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        mycursor = mydb.cursor()

        mycursor.execute("SELECT count(*) FROM ac_user where username=%s",(uname, ))
        cnt = mycursor.fetchone()[0]
        if cnt==0:
            mycursor.execute("SELECT max(id)+1 FROM ac_user")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1
            sql = "INSERT INTO ac_user(id,name,mobile, email, username,password,register_date) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            val = (maxid,name,mobile,email,uname,pass1,rdate)
            print(sql)
            mycursor.execute(sql, val)
            mydb.commit()            
            print(mycursor.rowcount, "record inserted.")
            msg='success'
            
            #if mycursor.rowcount==1:
            #    result="Registered Success"
            
        else:
            msg="fail"
    return render_template('web/register.html',msg=msg)

@app.route('/reg_doc',methods=['POST','GET'])
def reg_doc():
    msg=""
    act=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ac_department order by department")
    data1 = mycursor.fetchall()
    
    if request.method=='POST':
        
        name=request.form['name']
        dept=request.form['dept']
        
        mobile=request.form['mobile']
        email=request.form['email']
        hospital=request.form['hospital']
        location=request.form['location']
        
      
        now = datetime.datetime.now()
        rdate=now.strftime("%d-%m-%Y")
        

     
        mycursor.execute("SELECT max(id)+1 FROM ac_doctor")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        uname="D"+str(maxid)
        rn=randint(1000,9000)
        pass1=str(rn)
    
        sql = "INSERT INTO ac_doctor(id,name,dept,mobile,email,hospital,location,username,password,approved_status,register_date) VALUES (%s, %s, %s, %s, %s, %s, %s, %s,%s,%s,%s)"
        val = (maxid,name,dept,mobile,email,hospital,location,uname,pass1,'0',rdate)
        
        mycursor.execute(sql, val)
        mydb.commit()            
        #print(mycursor.rowcount, "record inserted.")
        msg='success'
        
        #if mycursor.rowcount==1:
        #    result="Registered Success"
     
    return render_template('web/reg_doc.html',msg=msg,data1=data1)



@app.route('/admin',methods=['POST','GET'])
def admin():
    msg=""
    email=""
    mess=""
    act=request.args.get("act")
    uname=""
    data=[]
   
    s1=""
    s2=""

    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ac_doctor")
    data = mycursor.fetchall()

    if act=="yes":
        aid=request.args.get("aid")
        mycursor.execute("SELECT * FROM ac_doctor where id=%s",(aid,))
        ds = mycursor.fetchone()
        email=ds[4]
        name=ds[1]
        
        docid=ds[7]
        pw=ds[8]
        mess="Dear "+name+", has Approved for Online Consultancy, Doctor ID:"+docid+", Password:"+pw
        
        mycursor.execute("update ac_doctor set approved_status=1,online_st=1 where id=%s",(aid,))
        mydb.commit()
        msg="ok"

    return render_template('admin.html',msg=msg,act=act,data=data,email=email,mess=mess)

@app.route('/add_dept', methods=['GET', 'POST'])
def add_dept():
    msg=""
    act=request.args.get("act")
    mycursor = mydb.cursor()
    if request.method=='POST':
        dept=request.form['dept']
     
        mycursor.execute("SELECT max(id)+1 FROM ac_department")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        sql = "INSERT INTO ac_department(id,department) VALUES (%s,%s)"
        val = (maxid,dept)
        mycursor.execute(sql, val)
        mydb.commit()
        
        msg="success"

    mycursor.execute("SELECT * FROM ac_department")
    data = mycursor.fetchall()

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from ac_department where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_dept'))
        
    return render_template('add_dept.html',msg=msg,act=act,data=data)


@app.route('/add_disease', methods=['GET', 'POST'])
def add_disease():
    msg=""
    act=request.args.get("act")
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM ac_department order by department")
    data1 = mycursor.fetchall()

    
    if request.method=='POST':
        dept=request.form['dept']
        disease=request.form['disease']
        symptom1=request.form['symptom1']
        symptom2=request.form['symptom2']
        symptom3=request.form['symptom3']
        symptom4=request.form['symptom4']
        symptom5=request.form['symptom5']
        test1=request.form['test1']
        test2=request.form['test2']
     
        mycursor.execute("SELECT max(id)+1 FROM ac_disease")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        sql = "INSERT INTO ac_disease(id,disease,symptom1,symptom2,symptom3,symptom4,symptom5,test1,test2) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        val = (maxid,disease,symptom1,symptom2,symptom3,symptom4,symptom5,test1,test2)
        mycursor.execute(sql, val)
        mydb.commit()
        
        msg="success"


        
    return render_template('add_disease.html',msg=msg,act=act,data1=data1)



@app.route('/add_data', methods=['GET', 'POST'])
def add_data():
    msg=""
    act=request.args.get("act")
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM ac_disease order by disease")
    data1 = mycursor.fetchall()

    
    if request.method=='POST':
        disease=request.form['disease']
        user_query=request.form['user_query']
        response1=request.form['response1']
     
        mycursor.execute("SELECT max(id)+1 FROM ac_data")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1

        sql = "INSERT INTO ac_data(id,user_query,response1,disease) VALUES (%s,%s,%s,%s)"
        val = (maxid,user_query,response1,disease)
        mycursor.execute(sql, val)
        mydb.commit()
        
        msg="success"

    mycursor.execute("SELECT * FROM ac_data where id>2")
    data = mycursor.fetchall()

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from ac_data where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('add_data'))
        
    return render_template('add_data.html',msg=msg,act=act,data=data,data1=data1)

@app.route('/edit_data', methods=['GET', 'POST'])
def edit_data():
    msg=""
    act=request.args.get("act")
    qid=request.args.get("qid")
    mycursor = mydb.cursor()

    
    if request.method=='POST':
        
        user_query=request.form['user_query']
        response1=request.form['response1']

        mycursor.execute("update ac_data set user_query=%s,response1=%s where id=%s",(user_query,response1,qid))
        mydb.commit()
        
        msg="success"

    mycursor.execute("SELECT * FROM ac_data where id=%s",(qid,))
    data = mycursor.fetchone()

    
        
    return render_template('edit_data.html',msg=msg,act=act,data=data)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    msg=""
    mycursor = mydb.cursor()
    if request.method=='POST':
        file = request.files['file']

        fn="datafile.csv"
        file.save(os.path.join("static/upload", fn))

        filename = 'static/upload/datafile.csv'
        data1 = pd.read_csv(filename, header=0)
        data2 = list(data1.values.flatten())
        for ss in data1.values:

            mycursor.execute("SELECT max(id)+1 FROM ac_disease")
            maxid = mycursor.fetchone()[0]
            if maxid is None:
                maxid=1

            

            sql = "INSERT INTO ac_disease(id,disease,symptom1,symptom2,symptom3,symptom4,symptom5,test1,test2,consultant) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
            val = (maxid,ss[0],ss[1],ss[2],ss[3],ss[4],ss[5],ss[6],ss[7],ss[8])
            #mycursor.execute(sql, val)
            #mydb.commit()
        
        msg="success"


    return render_template('upload.html',msg=msg)


@app.route('/process', methods=['GET', 'POST'])
def process():
    msg=""
    cnt=0
    

    filename = 'static/upload/datafile.csv'
    data1 = pd.read_csv(filename, header=0)
    data2 = list(data1.values.flatten())

    
    data=[]
    i=0
    sd=len(data1)
    rows=len(data1.values)
    
    #print(str(sd)+" "+str(rows))
    for ss in data1.values:
        cnt=len(ss)
        data.append(ss)
    cols=cnt

    
    return render_template('process.html',data=data, msg=msg, rows=rows, cols=cols)

@app.route('/process2', methods=['GET', 'POST'])
def process2():
    msg=""
    act=request.args.get("act")
    
    return render_template('process2.html',msg=msg, act=act)

@app.route('/view_data1', methods=['GET', 'POST'])
def view_data1():
    msg=""
    act=request.args.get("act")
  
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM ac_disease")
    data = mycursor.fetchall()   

    if act=="del":
        did=request.args.get("did")
        mycursor.execute("delete from ac_disease where id=%s",(did,))
        mydb.commit()
        return redirect(url_for('view_data1'))
        
    
    return render_template('view_data1.html',msg=msg,act=act,data=data)

@app.route('/view_user', methods=['GET', 'POST'])
def view_user():
    msg=""
    act=request.args.get("act")
  
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM ac_user")
    data = mycursor.fetchall()   
    
    return render_template('view_user.html',msg=msg,act=act,data=data)


@app.route('/userhome',methods=['POST','GET'])
def userhome():
    msg=""
    uname=""
    if 'username' in session:
        uname = session['username']
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ac_user where username=%s",(uname,))
    data = mycursor.fetchone()

    
    mycursor.execute("update ac_data set status=0,response2='',qno=0")
    mydb.commit()
    
    return render_template('userhome.html',msg=msg,data=data)

def getImagesAndLabels(path):

    
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        #mm=imagePath.split('.')
        #id=mm[0]+"."+mm[1]
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

@app.route('/add_photo',methods=['POST','GET'])
def add_photo():
    uname=""
    view = request.args.get('view')
    vid = request.args.get('vid')
    if 'username' in session:
        uname = session['username']
    
    ff1=open("photo.txt","w")
    ff1.write("2")
    ff1.close()

    ff1=open("view.txt","w")
    ff1.write(view)
    ff1.close()

    #ff2=open("mask.txt","w")
    #ff2.write("face")
    #ff2.close()
    act = request.args.get('act')

    cursor = mydb.cursor()
    
    cursor.execute("SELECT * FROM ac_user where id=%s",(vid,))
    value = cursor.fetchone()
    name=value[1]
    
    ff=open("user.txt","w")
    ff.write(name)
    ff.close()

    ff=open("user1.txt","w")
    ff.write(vid)
    ff.close()
    

    
    
    if request.method=='POST':
        vid=request.form['vid']
        fimg="v"+vid+".jpg"
        

        cursor.execute('delete from ac_face WHERE vid = %s', (vid, ))
        mydb.commit()

        

        ff=open("det.txt","r")
        v=ff.read()
        ff.close()
        vv=int(v)
        v1=vv-1
        vd=0
        if view=='c':
            vd=int(vid)+100
        elif view=='b':
            vd=int(vid)+50
        else:
            vd=int(vid)
        
        vface1="User"+view+"."+str(vd)+"."+str(v1)+".jpg"
        
        i=2
        while i<vv:
            
            cursor.execute("SELECT max(id)+1 FROM ac_face")
            maxid = cursor.fetchone()[0]
            if maxid is None:
                maxid=1
            vface="User"+view+"."+str(vd)+"."+str(i)+".jpg"
            sql = "INSERT INTO ac_face(id, vid, vface) VALUES (%s, %s, %s)"
            val = (maxid, vid, vface)
            print(val)
            cursor.execute(sql,val)
            mydb.commit()
            i+=1

        
        if view=="a":
            cursor.execute('update ac_user set fimg=%s WHERE id = %s', (vface1, vid))
            mydb.commit()
        if view=="b":
            cursor.execute('update ac_user set fimg2=%s WHERE id = %s', (vface1, vid))
            mydb.commit()
        if view=="c":
            cursor.execute('update ac_user set fimg3=%s WHERE id = %s', (vface1, vid))
            mydb.commit()
        shutil.copy('static/faces/f1.jpg', 'static/photo/'+vface1)

        
        ##########
        
        ##Training face
        # Path for face image database
        path = 'dataset'

        recognizer = cv2.face.LBPHFaceRecognizer_create()

        # function to get the images and label data
        

        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))

        # Save the model into trainer/trainer.yml
        tname="trainer"+vid+".yml"
        recognizer.write('trainer/'+tname) # recognizer.save() worked on Mac, but not on Pi

        # Print the numer of faces trained and end program
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))






        #################################################
        cursor = mydb.cursor()
        cursor.execute("SELECT * FROM ac_face where vid=%s",(vid, ))
        dt = cursor.fetchall()
        for rs in dt:
            ##Preprocess
            path="static/frame/"+rs[2]
            path2="static/process1/"+rs[2]
            mm2 = PIL.Image.open(path).convert('L')
            rz = mm2.resize((200,200), PIL.Image.ANTIALIAS)
            rz.save(path2)
            
            '''img = cv2.imread(path2) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            path3="static/process2/"+rs[2]
            cv2.imwrite(path3, dst)'''
            #noice
            img = cv2.imread('static/process1/'+rs[2]) 
            dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
            fname2='ns_'+rs[2]
            cv2.imwrite("static/process1/"+fname2, dst)
            ######
            ##bin
            image = cv2.imread('static/process1/'+rs[2])
            original = image.copy()
            kmeans = kmeans_color_quantization(image, clusters=4)

            # Convert to grayscale, Gaussian blur, adaptive threshold
            gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3,3), 0)
            thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)
            
            # Draw largest enclosing circle onto a mask
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                ((x, y), r) = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
                cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
                break
            
            # Bitwise-and for result
            result = cv2.bitwise_and(original, original, mask=mask)
            result[mask==0] = (0,0,0)

            
            ###cv2.imshow('thresh', thresh)
            ###cv2.imshow('result', result)
            ###cv2.imshow('mask', mask)
            ###cv2.imshow('kmeans', kmeans)
            ###cv2.imshow('image', image)
            ###cv2.waitKey()

            #cv2.imwrite("static/process1/bin_"+rs[2], thresh)
            

            ###RPN - Segment
            img = cv2.imread('static/process1/'+rs[2])
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/fg_"+rs[2]
            segment.save(path3)
            ####
            img = cv2.imread('static/process2/fg_'+rs[2])
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/fg_"+rs[2]
            segment.save(path3)
            '''
            img = cv2.imread(path2)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)

            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            segment = cv2.subtract(sure_bg,sure_fg)
            img = Image.fromarray(img)
            segment = Image.fromarray(segment)
            path3="static/process2/"+rs[2]
            segment.save(path3)
            '''
            #####
            image = cv2.imread(path2)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edged = cv2.Canny(gray, 50, 100)
            image = Image.fromarray(image)
            edged = Image.fromarray(edged)
            path4="static/process3/"+rs[2]
            edged.save(path4)
            ##
        ###
        cursor.execute("SELECT count(*) FROM ac_face where vid=%s",(vid, ))
        cnt = cursor.fetchone()[0]
        
        return redirect(url_for('view_photo',vid=vid,act='success'))
        
    
    cursor.execute("SELECT * FROM ac_user where username=%s",(uname,))
    data = cursor.fetchone()
    return render_template('add_photo.html',data=data, vid=vid, view=view)

def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

def CNN():
    #Lets start by loading the Cifar10 data
    (X, y), (X_test, y_test) = cifar10.load_data()

    #Keep in mind the images are in RGB
    #So we can normalise the data by diving by 255
    #The data is in integers therefore we need to convert them to float first
    X, X_test = X.astype('float32')/255.0, X_test.astype('float32')/255.0

    #Then we convert the y values into one-hot vectors
    #The cifar10 has only 10 classes, thats is why we specify a one-hot
    #vector of width/class 10
    y, y_test = u.to_categorical(y, 10), u.to_categorical(y_test, 10)

    #Now we can go ahead and create our Convolution model
    model = Sequential()
    #We want to output 32 features maps. The kernel size is going to be
    #3x3 and we specify our input shape to be 32x32 with 3 channels
    #Padding=same means we want the same dimensional output as input
    #activation specifies the activation function
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same',
                     activation='relu'))
    #20% of the nodes are set to 0
    model.add(Dropout(0.2))
    #now we add another convolution layer, again with a 3x3 kernel
    #This time our padding=valid this means that the output dimension can
    #take any form
    model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
    #maxpool with a kernet of 2x2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #In a convolution NN, we neet to flatten our data before we can
    #input it into the ouput/dense layer
    model.add(Flatten())
    #Dense layer with 512 hidden units
    model.add(Dense(512, activation='relu'))
    #this time we set 30% of the nodes to 0 to minimize overfitting
    model.add(Dropout(0.3))
    #Finally the output dense layer with 10 hidden units corresponding to
    #our 10 classe
    model.add(Dense(10, activation='softmax'))
    #Few simple configurations
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])
    #Run the algorithm!
    model.fit(X, y, validation_data=(X_test, y_test), epochs=25,
              batch_size=512)
    #Save the weights to use for later
    model.save_weights("cifar10.hdf5")
    #Finally print the accuracy of our model!
    print("Accuracy: &2.f%%" %(model.evaluate(X_test, y_test)[1]*100))

@app.route('/view_photo',methods=['POST','GET'])
def view_photo():
    uname=""
    if 'username' in session:
        uname = session['username']

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ac_user where username=%s",(uname,))
    data = mycursor.fetchone()
    
    ff1=open("photo.txt","w")
    ff1.write("1")
    ff1.close()
    vid=""
    value=[]
    if request.method=='GET':
        vid = request.args.get('vid')
        
        mycursor.execute("SELECT * FROM ac_face where vid=%s",(vid, ))
        value = mycursor.fetchall()

    if request.method=='POST':
        print("Training")
        vid=request.form['vid']
        
        #shutil.copy('static/img/11.png', 'static/process4/'+rs[2])
       
        #return redirect(url_for('view_photo1',vid=vid))
        
    return render_template('view_photo.html', data=data,result=value,vid=vid)


@app.route('/user_test',methods=['POST','GET'])
def user_test():
    msg=""
    uname=""
    act=request.args.get("act")
    #if 'username' in session:
    #    uname = session['username']

    ff=open("uname.txt","r")
    uname=ff.read()
    ff.close()

    ff=open("emotion.txt","r")
    emo=ff.read()
    ff.close()
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ac_user where username=%s",(uname,))
    data = mycursor.fetchone()

    return render_template('user_test.html', act=act,data=data,emo=emo)

@app.route('/pro',methods=['POST','GET'])
def pro():
    msg=""

    act=request.args.get("act")
    st=""
    ff=open("emotion.txt","r")
    emo=ff.read()
    ff.close()

    #emo="Mild"

    if emo=="":
        s=1
    else:
        st="1"

    return render_template('pro.html', st=st)


@app.route('/pro1',methods=['POST','GET'])
def pro1():
    msg=""

    act=request.args.get("act")
    st=""
    ff=open("emotion.txt","r")
    emo=ff.read()
    ff.close()

    mess="Pain predicted, severity level is "+emo
    #speak(mess)

    
    return render_template('pro1.html', emo=emo,mess=mess)

@app.route('/page1',methods=['POST','GET'])
def page1():
    msg=""
    st=""
    ff=open("emotion.txt","r")
    emo=ff.read()
    ff.close()

    emo="Mild"

    if emo=="":
        s=1
    else:
        st="1"
    
    return render_template('page1.html', st=st)

##
@app.route('/page2',methods=['POST','GET'])
def page2():
    msg=""
    qry=""
    qid=""
    qno=""
    uname=""
    consult=""
    row=[]
    ydata=[]
    qq1=0
    s1=""
    #if 'username' in session:
    #    uname = session['username']
    ff=open("uname.txt","r")
    uname=ff.read()
    ff.close()

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ac_user where username=%s",(uname,))
    data = mycursor.fetchone()
    
    ff=open("emotion.txt","r")
    emo=ff.read()
    ff.close()
        
    if request.method=='POST':
        qry=request.form['msg_input']
        qid1=request.form['qid1']

        try:
            mycursor.execute("SELECT * FROM ac_data where status>0 && qno>0 order by qno")
            ydata = mycursor.fetchall()
            ##
            stemmer = PorterStemmer()
        
            from wordcloud import STOPWORDS
            STOPWORDS.update(['rt', 'mkr', 'didn', 'bc', 'n', 'm', 
                              'im', 'll', 'y', 've', 'u', 'ur', 'don', 
                              'p', 't', 's', 'aren', 'kp', 'o', 'kat', 
                              'de', 're', 'amp', 'will'])

            def lower(text):
                return text.lower()

            def remove_specChar(text):
                return re.sub("#[A-Za-z0-9_]+", ' ', text)

            def remove_link(text):
                return re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+', ' ', text)

            def remove_stopwords(text):
                return " ".join([word for word in 
                                 str(text).split() if word not in STOPWORDS])

            def stemming(text):
                return " ".join([stemmer.stem(word) for word in text.split()])

            def cleanTxt(text):
                text = lower(text)
                text = remove_specChar(text)
                text = remove_link(text)
                #text = remove_stopwords(text)
                #text = stemming(text)
                
                return text
            ##
            clean_msg=cleanTxt(qry)
            #print(clean_msg)
            ###
            y=0
            mycursor.execute("SELECT count(*) FROM ac_data where status=1")
            cn1 = mycursor.fetchone()[0]
            if cn1>=2:
                mycursor.execute("SELECT disease,count(*) FROM ac_data where disease!='' && status=1 group by disease")
                rw1 = mycursor.fetchall()
                for rw11 in rw1:
                    if rw11[1]>=2:
                        y=1
                        s1="1"
                        ds=rw11[0]
                        print(ds)
                        mycursor.execute("SELECT * FROM ac_disease where disease=%s",(ds,))
                        rw2 = mycursor.fetchone()
                        consult=rw2[9]
                        break
                if s1=="1":
                    mm="Please consult the "+consult
                    
                    docid=""
                    mycursor.execute("SELECT * FROM ac_doctor where dept=%s && online_st=1",(consult,))
                    d5 = mycursor.fetchall()
                    for d55 in d5:
                        docid=d55[7]

                    fnn=docid+".txt"
                    print(fnn)
                    ff=open("static/"+fnn,"w")
                    ff.write(uname)
                    ff.close()
                    speak(mm)

            ##
            qcc=0
            mycursor.execute("SELECT * FROM ac_data order by id")
            crr = mycursor.fetchall()
            for crr1 in crr:
                qcc=crr1[0]
            qcc1=str(qcc)
            ###
            qq=0
            if qid1=="1" or qcc1==qid1:
                qq=2
            else:
                qq1=int(qid1)+1
                mycursor.execute("SELECT * FROM ac_data where id=%s",(qq1,))
                r1 = mycursor.fetchone()
                qq=r1[0]
                su=0
                x=0
                clean_msg=clean_msg.strip()
                if clean_msg=='yes' or clean_msg=='yes yes':
                    su=1
                elif clean_msg=='no' or clean_msg=='no no':
                    su=2
                else:
                    '''c1=clean_msg.split(" ")
                    for c11 in c1:
                        if c11=="yes" or c11=="no":
                            s=1
                        elif c11 in clean_msg:
                            x+=1
                            v1='%'+c11+'%'
                            '''
                    v1='%'+clean_msg+'%'
                    mycursor.execute("SELECT count(*) FROM ac_data where response1 like %s",(v1,))
                    nn = mycursor.fetchone()[0]
                    if nn>0:                    
                        mycursor.execute("SELECT * FROM ac_data where response1 like %s",(v1,))
                        r2 = mycursor.fetchone()
                        qq=r2[0]
                        su=1

                mycursor.execute("SELECT max(qno)+1 FROM ac_data")
                maxid2 = mycursor.fetchone()[0]
                if maxid2 is None:
                    maxid2=1
                print(clean_msg)
                print("qno")    
                print(maxid2)
                print("su")
                print(su)
                mycursor.execute("update ac_data set response2=%s,status=%s,qno=%s where id=%s",(clean_msg,su,maxid2,qid1))
                mydb.commit()

                ##
                mycursor.execute("SELECT * FROM ac_data where id=%s",(qid1,))
                r3 = mycursor.fetchone()
                quest=r3[1]
                mycursor.execute("SELECT max(id)+1 FROM ac_temp")
                maxid = mycursor.fetchone()[0]
                if maxid is None:
                    maxid=1

                sql = "INSERT INTO ac_temp(id,question,answer,uname) VALUES (%s,%s,%s,%s)"
                val = (maxid,quest,clean_msg,uname)
                mycursor.execute(sql, val)
                mydb.commit()
                ##
            print("qq")
            print(qq)
            mycursor.execute("SELECT * FROM ac_data where id=%s",(qq,))
            row = mycursor.fetchone()
            
            qid=str(row[0])
            qsn=row[1]
            '''if qsn=="":
                mycursor.execute("SELECT * FROM ac_data where id=1")
                row = mycursor.fetchone()
                speak(row[1])
            else:
                speak(qsn)'''

            
        except:
            print("try")
        
    else:
        mycursor.execute("SELECT * FROM ac_data where id=1")
        row = mycursor.fetchone()
        qid=str(row[0])
        qsn=row[1]
        #speak(qsn)

    return render_template('page2.html', data=data,row=row,qid=qid,s1=s1,consult=consult,ydata=ydata,emo=emo)

###################
@app.route('/view_consult',methods=['POST','GET'])
def view_consult():
    msg=""
    consult=request.args.get("consult")
    docid=request.args.get("docid")
    uname=""
    s1=""
    act=request.args.get("act")
    if 'username' in session:
        uname = session['username']
    #print(consult)
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ac_doctor where dept=%s && online_st=1",(consult,))
    data = mycursor.fetchall()

    if act=="ok":
        
        mycursor.execute("update ac_user set docid=%s where username=%s",(docid,uname))
        mydb.commit()
        msg="ok"
        

    return render_template('view_consult.html',msg=msg,data=data,docid=docid)

@app.route('/meet',methods=['POST','GET'])
def meet():
    msg=""
    consult=request.args.get("consult")
    
    uname=""
    doctor=""
    s1=""
    #if 'username' in session:
    #    uname = session['username']
    ff=open("uname.txt","r")
    uname=ff.read()
    ff.close()

    ff=open("emotion.txt","r")
    emo=ff.read()
    ff.close()
   
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ac_user where username=%s",(uname,))
    data = mycursor.fetchone()
    name=data[1]
    print(name)
    
    mycursor.execute("SELECT * FROM ac_doctor where dept=%s order by rand()",(consult,))
    data1 = mycursor.fetchall()
    print(data1)
    docid=""
    for ds in data1:
        doctor=ds[1]
        docid=ds[7]

    rn=randint(100,999)
    fn="f"+str(rn)+".jpg"
    shutil.copy("static/faces/f1.jpg","static/detect/"+fn)
    mycursor.execute("update ac_user set docid=%s,emotion=%s,detect_face=%s where username=%s",(docid,emo,fn,uname))
    mydb.commit()

    return render_template('meet.html', doctor=doctor,data=data,name=name)

@app.route('/meetapi',methods=['POST','GET'])
def meetapi():
    msg=""
    name=request.args.get("name")
    return render_template('meetapi.html', msg=msg,name=name)

@app.route('/test2',methods=['POST','GET'])
def test2():
    msg=""
    name=request.args.get("name")
    return render_template('test2.html', msg=msg,name=name)


@app.route('/store_data', methods=['POST'])
def store_data():
    data = request.json  # Get data from AJAX request
    data_list.append(data)  # Store data in the list

    
    return jsonify({'message': 'Data stored successfully'})

@app.route('/get_data', methods=['GET'])
def get_data():
    input_value = request.args.get('input_value')

    # Query the database cursor(dictionary=True)
    mycursor = mydb.cursor()
    dd='%'+input_value+'%'
    mycursor.execute("SELECT * FROM ac_disease where disease like %s",(dd,))
    result = mycursor.fetchall()

    return jsonify(result)


@app.route('/test1',methods=['POST','GET'])
def test1():
    msg=""
    data=[]


    return render_template('test1.html', data=data)   

@app.route('/bot1', methods=['GET'])
def bot1():
    
    mycursor = mydb.cursor()
    
    mycursor.execute("SELECT * FROM ac_temp")
    result = mycursor.fetchall()

    return jsonify(result)

@app.route('/bot',methods=['POST','GET'])
def bot():
    msg=""
    data=[]


    return render_template('bot.html', data=data)   


def speak(audio):
    engine = pyttsx3.init()
    engine.say(audio)
    engine.runAndWait()


@app.route('/doc_home',methods=['POST','GET'])
def doc_home():
    msg=""
    uname=""
    data2=[]
    st=""
    act=request.args.get("act")
    #if 'username' in session:
    #    uname = session['username']
    ff=open("doc.txt","r")
    uname=ff.read()
    ff.close()
        
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ac_doctor where username=%s",(uname,))
    data = mycursor.fetchone()

    if act=="1":
        mycursor.execute("update ac_doctor set online_st=1 where username=%s",(uname,))
        mydb.commit()
        return redirect(url_for('doc_home'))

    if act=="2":
        mycursor.execute("update ac_doctor set online_st=0 where username=%s",(uname,))
        mydb.commit()
        return redirect(url_for('doc_home'))

    mycursor.execute("SELECT count(*) FROM ac_user where docid=%s",(uname,))
    cnt = mycursor.fetchone()[0]
    if cnt>0:
        st="1"
        mycursor.execute("SELECT * FROM ac_user where docid=%s",(uname,))
        dat = mycursor.fetchall()
        for ds in dat:
            dt=[]
            dt.append(ds[0])
            dt.append(ds[1])
            dt.append(ds[2])
            dt.append(ds[3])
            dt.append(ds[4])
            dt.append(ds[5])
            dt.append(ds[6])
            dt.append(ds[7])
            dt.append(ds[8])
            dt.append(ds[9])
            dt.append(ds[10])
            dt.append(ds[11])
            dt.append(ds[12])

            dt1=[]
            mycursor.execute("SELECT * FROM ac_temp where uname=%s",(ds[4],))
            dat2 = mycursor.fetchall()
            for ds2 in dat2:
                dt2=[]
                dt2.append(ds2[0])
                dt2.append(ds2[1])
                dt2.append(ds2[2])
                dt2.append(ds2[3])
                dt1.append(dt2)

            dt.append(dt1)
            data2.append(dt)
        
        
    return render_template('doc_home.html',msg=msg,data=data,data2=data2,st=st)

@app.route('/doc1',methods=['POST','GET'])
def doc1():
    msg=""
    uname=""
    data2=[]
    user=""
    st=""
    act=request.args.get("act")
    #if 'username' in session:
    #    uname = session['username']
    ff=open("doc.txt","r")
    uname=ff.read()
    ff.close()

    fnn=uname+".txt"
    ff=open("static/"+fnn,"r")
    stt=ff.read()
    ff.close()

    if stt=="":
        st=""
    else:
        st="1"

    user=stt

    return render_template('doc1.html',msg=msg,st=st,user=user)


@app.route('/doc_meet',methods=['POST','GET'])
def doc_meet():
    msg=""
    uname=""
    data2=[]
    user=request.args.get("user")
    act=request.args.get("act")
    #if 'username' in session:
    #    uname = session['username']
    ff=open("doc.txt","r")
    uname=ff.read()
    ff.close()

    fnn=uname+".txt"
    ff=open("static/"+fnn,"r")
    stt=ff.read()
    ff.close()

    user=stt
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ac_doctor where username=%s",(uname,))
    data = mycursor.fetchone()
    name=data[1]

    mycursor.execute("SELECT count(*) FROM ac_user where username=%s && docid=%s",(user,uname))
    cnt = mycursor.fetchone()[0]
    if cnt>0:
        st="1"
        mycursor.execute("SELECT * FROM ac_user where username=%s && docid=%s",(user,uname))
        dat = mycursor.fetchall()
        for ds in dat:
            dt=[]
            dt.append(ds[0])
            dt.append(ds[1])
            dt.append(ds[2])
            dt.append(ds[3])
            dt.append(ds[4])
            dt.append(ds[5])
            dt.append(ds[6])
            dt.append(ds[7])
            dt.append(ds[8])
            dt.append(ds[9])
            dt.append(ds[10])
            dt.append(ds[11])
            dt.append(ds[12])

            dt1=[]
            mycursor.execute("SELECT * FROM ac_temp where uname=%s",(ds[4],))
            dat2 = mycursor.fetchall()
            for ds2 in dat2:
                dt2=[]
                dt2.append(ds2[0])
                dt2.append(ds2[1])
                dt2.append(ds2[2])
                dt2.append(ds2[3])
                dt1.append(dt2)

            dt.append(dt1)
            data2.append(dt)

      
    return render_template('doc_meet.html',msg=msg,data=data,data2=data2,name=name)

@app.route('/meet2',methods=['POST','GET'])
def meet2():
    msg=""
    uname=""
    user=request.args.get("user")
    act=request.args.get("act")
    #if 'username' in session:
    #    uname = session['username']
    ff=open("doc.txt","r")
    uname=ff.read()
    ff.close()
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ac_doctor where username=%s",(uname,))
    data = mycursor.fetchone()
    name=data[1]

    mycursor.execute("SELECT * FROM ac_user where username=%s",(user,))
    data2 = mycursor.fetchone()

   
        
    return render_template('meet2.html',msg=msg,data=data,data2=data2,name=name)

@app.route('/logout2')
def logout2():
    uname=""
    if 'username' in session:
        uname = session['username']

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM ac_doctor where username=%s",(uname,))
    data = mycursor.fetchone()

    mycursor.execute("update ac_user set docid='' where docid=%s",(uname,))
    mydb.commit()
    
    # remove the username from the session if it is there
    #session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    #session.pop('username', None)
    return redirect(url_for('index'))
###############
def gen(camera):
    
    while True:
        frame = camera.get_frame()


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():

    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
###############
def gen2(camera):
    
    while True:
        frame = camera.get_frame()


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed2')
def video_feed2():

    return Response(gen2(VideoCamera2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
