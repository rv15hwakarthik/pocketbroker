# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 17:42:38 2018

@author: Vishwakarthik Ramesh
"""
#Machine Learning
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Minor_Project.csv")
X=dataset.iloc[:,3:-1]
Y=dataset.iloc[:,-1]

#converting yes and no to numbers
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X.iloc[:,10]=labelencoder_X.fit_transform(X.iloc[:,10])
X.iloc[:,9]=labelencoder_X.fit_transform(X.iloc[:,9])
X.iloc[:,11]=labelencoder_X.fit_transform(X.iloc[:,11])
onehotencoder=OneHotEncoder(categorical_features=[9,10,11])
X=onehotencoder.fit_transform(X).toarray()

#removing the dummy variable problem
X=X[:,[0,2,4,6,7,8,9,10,11,12,13,14,15,16,17]]

#splitting the dataset into training and testing set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#fitting multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,Y)

#predicting the result
#y_pred = regressor.predict(X_test)

#plotting the graph for comparison
"""x_axis=np.array([0,1,2,3,4,5,6,7,8,9])
plt.scatter(x_axis,Y_test,color='red')
plt.scatter(x_axis,y_pred,color='green')
plt.title('Comparison between Real and Predicted Brokerage')
plt.xlabel('Index')
plt.ylabel('Brokerage')
plt.show()"""

#GUI
from tkinter import *
root = Tk()
root.title("Pocket-Broker")
root.maxsize(width=700,height=790)
root.minsize(width=700,height=790)

frame1 = Frame(root)
frame1.pack()
canvas=Canvas(root,width=600,height=50)
canvas.pack()
line = canvas.create_line(0,25,600,25)   # starting x,starting y,ending x,ending y# line to separate the frames
frame2 = Frame(root)
frame2.pack()
canvas=Canvas(root,width=600,height=50)
canvas.pack()
line = canvas.create_line(0,25,600,25)
frame3 = Frame(root)
frame3.pack()
frame3_2 = Frame(root)
frame3_2.pack()
canvas=Canvas(root,width=600,height=50)
canvas.pack()
line = canvas.create_line(0,25,600,25)
frame4 = Frame(root)
frame4.pack()
frame4_2 =Frame(root)
frame4_2.pack()
canvas=Canvas(root,width=600,height=50)
canvas.pack()
line = canvas.create_line(0,25,600,25)
frame5 = Frame(root)
frame5.pack()
frame5_2 =Frame(root)
frame5_2.pack()
canvas=Canvas(root,width=600,height=50)
canvas.pack()
line = canvas.create_line(0,25,600,25)
frame6 = Frame(root)
frame6.pack()

# ***Frame 1***
empty_label = Label(frame1)
empty_label.pack()

label_header=Label(frame1,text="Pocket-Broker",font=('arial',20,'bold'))
label_header.pack()

# ***Frame 2***
label_type=Label(frame2,text="Type of Flat:(2.5/2.75/3)")
label_type.pack(side=LEFT)

entry_type=Entry(frame2)
entry_type.insert(3,"3")
entry_type.pack(side=LEFT)

# ***Frame 3***
frame3_header= Label(frame3,text='Electronics',font=('arial',10,'bold'))
frame3_header.pack()

empty_label = Label(frame3)
empty_label.pack()

label_ac=Label(frame3_2,text="No. of ACs")
label_ac.grid(sticky="W",row=0)

entry_ac=Entry(frame3_2)
entry_ac.insert(0,"0")
entry_ac.grid(sticky="W",row=0,column=1)

label_acw=Label(frame3_2,text="No. of ACs in Warranty")
label_acw.grid(sticky="W",row=1)

entry_acw=Entry(frame3_2)
entry_acw.insert(0,"0")
entry_acw.grid(sticky="W",row=1,column=1)

label_fridge=Label(frame3_2,text="No. of Fridges")
label_fridge.grid(sticky="W",row=2)

entry_fridge=Entry(frame3_2)
entry_fridge.insert(0,"0")
entry_fridge.grid(sticky="W",row=2,column=1)

label_fridgew=Label(frame3_2,text="No. of Fridges in Warranty")
label_fridgew.grid(sticky="W",row=3)

entry_fridgew=Entry(frame3_2)
entry_fridgew.insert(0,"0")
entry_fridgew.grid(sticky="W",row=3,column=1)

label_wm=Label(frame3_2,text="No. of Washing Machines")
label_wm.grid(sticky="W",row=4)

entry_wm=Entry(frame3_2)
entry_wm.insert(0,"0")
entry_wm.grid(sticky="W",row=4,column=1)

label_wmw=Label(frame3_2,text="No. of Washing Machines in Warranty")
label_wmw.grid(sticky="W",row=5)

entry_wmw=Entry(frame3_2)
entry_wmw.insert(0,"0")
entry_wmw.grid(sticky="W",row=5,column=1)

label_tv=Label(frame3_2,text="No. of TVs")
label_tv.grid(sticky="W",row=6)

entry_tv=Entry(frame3_2)
entry_tv.insert(0,"0")
entry_tv.grid(sticky="W",row=6,column=1)

label_tvw=Label(frame3_2,text="No. of TVs in Warranty")
label_tvw.grid(sticky="W",row=7)

entry_tvw=Entry(frame3_2)
entry_tvw.insert(0,"0")
entry_tvw.grid(sticky="W",row=7,column=1)

# ***Frame 4***
frame4_header= Label(frame4,text='Kitchen Appliances',font=('arial',10,'bold'))
frame4_header.pack()

empty_label = Label(frame4)
empty_label.pack()


label_micro = Label(frame4_2,text=' Microwave(YES/NO)')
label_micro.grid(row=0,sticky="W")
entry_micro = Entry(frame4_2,width=10)
entry_micro.insert(0,'NO') #setting default value and place holder
entry_micro.grid(row=0,column=1,sticky="W")

label_chimney = Label(frame4_2,text=' Chimney(YES/NO)')
label_chimney.grid(row=0,column=3,sticky="W")
entry_chimney = Entry(frame4_2,width=10)
entry_chimney.insert(0,'NO')
entry_chimney.grid(row=0,column=4,sticky="W")

label_utensils = Label(frame4_2,text=' Utensils(YES/NO)')
label_utensils.grid(row=0,column=6,sticky="W")
entry_utensils = Entry(frame4_2,width=10)
entry_utensils.insert(1,'YES')
entry_utensils.grid(row=0,column=7,sticky="W")

# ***Frame 5***
frame4_header= Label(frame5,text='Furniture',font=('arial',10,'bold'))
frame4_header.pack()

empty_label = Label(frame5)
empty_label.pack()


label_beds = Label(frame5_2,text=' No. of Beds')
label_beds.grid(row=0,sticky="W")
entry_beds = Entry(frame5_2,width=10)
entry_beds.insert(0,'0') #setting default value and place holder
entry_beds.grid(row=0,column=1,sticky="W")

label_tables = Label(frame5_2,text=' No. of Tables')
label_tables.grid(row=0,column=3,sticky="W")
entry_tables = Entry(frame5_2,width=10)
entry_tables.insert(0,'0')
entry_tables.grid(row=0,column=4,sticky="W")

label_chairs = Label(frame5_2,text=' No. of Chairs')
label_chairs.grid(row=0,column=6,sticky="W")
entry_chairs = Entry(frame5_2,width=10)
entry_chairs.insert(0,'0')
entry_chairs.grid(row=0,column=7,sticky="W")

# ***Frame 6***

output_label1 = Label(frame6)
output_label1.pack(side=BOTTOM)

def ml_algorithm():
    type_=float(entry_type.get())
    acs=int(entry_ac.get())
    acsw=int(entry_acw.get())
    fridge=int(entry_fridge.get())
    fridgew=int(entry_fridgew.get())
    wm=int(entry_wm.get())
    wmw=int(entry_wmw.get())
    tv=int(entry_tv.get())
    tvw=int(entry_tvw.get())
    micro=entry_micro.get()
    if micro=="YES" or "Yes" or "yes":
        micro=1
    else:
        micro=0
    chimney=entry_chimney.get()
    if chimney=="YES" or "Yes" or "yes":
        chimney=1
    else:
        chimney=0
    utensils=entry_utensils.get()
    if utensils=="YES" or "Yes" or "yes":
        utensils=1
    else:
        utensils=0
    beds=int(entry_beds.get())
    tables=int(entry_tables.get())
    chairs=int(entry_chairs.get())
    
    user_input=np.array([[micro,chimney,utensils,type_,acs,acsw,fridge,fridgew,wm,wmw,tv,tvw,beds,tables,chairs]])
    prediction=regressor.predict(user_input)
    prediction=int(prediction)
    if prediction<0:
        output_label1.config(text='Predicted Brokerage = 1000')
    else:
        output_label1.config(text='Predicted Brokerage ='+str(prediction))

button = Button(frame6,text='Predict',command=ml_algorithm)
button.pack()

empty_label=Label(frame6)
empty_label.pack()

#Ending 
root.mainloop()
