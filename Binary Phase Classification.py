#Import packages
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.utils import class_weight
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

#Importing the data (v2)
order = 3  # this is an order of approximation
modes = 15 # the number of modes
mode_val=[]
pump_poww=[]
for w0 in np.arange(570.0,600.5,0.25): # w0 is cut-off wavelength in nm
    alldata=np.load(r"Data_file_path".format(w0,modes,order),encoding='bytes',allow_pickle=True).item(0)
    mode_val.append(list(alldata.values())[1])
    pump_poww.append(list(alldata.values())[0])
training_w0=[]
training_pump=[]

for i in range(122):
    if len(mode_val[i]) == 129:
        for j in range(129):
            training_w0.append(np.arange(570.0,600.5,0.25)[i])
            training_pump.append(pump_poww[i][j])

#Define the threshold to classify condensed modes, change as necessary
threshold=1e6
y=[]

# Find output for 40 w and 33 pump_pow
for w0 in range(122):
    w=[]
    for pump_power in range(129):
        
        if len(mode_val[w0]) == 129:
            populated_modes=[0 for i in range(len(mode_val[w0][pump_power]))]
            for mode in range(len(mode_val[w0][pump_power])):
                if mode_val[w0][pump_power][mode]>=threshold :
                    populated_modes[mode]=1
            y.append(populated_modes)

#Putting Data into the Pandas dataframe form
x=[]
x.append(training_w0)
x.append(training_pump)
x=pd.DataFrame(data=x)

X=x.T
X=pd.DataFrame.to_numpy(X)
y=pd.DataFrame(y)
y=pd.DataFrame.to_numpy(y)

#Splitting the data into test and train.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Normalising p_p and w0
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the neural network using the training data
model = Sequential()
model.add(Dense(2, input_dim=2, activation='relu'))
model.add(Dense(49, activation='relu'))
model.add(Dense(49, activation='relu'))
model.add(Dense(49, activation='relu'))
model.add(Dense(15, activation='sigmoid'))

# Compile the network; use adam optimiser and binary_crossentropy loss function
model.compile(loss='binary_crossentropy',
              optimizer='adam')

# Train the model with 10 epochs, a batch size of 100 and balanced class weights
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)
model.fit(X_train, y_train, epochs=10, batch_size=100, class_weight=class_weights)

preds = model.predict(X_test)
preds[preds>=0.5] = 1
preds[preds<0.5] = 0

# Define a function to convert binary strings to numbers
def convert(list): 
    lis = [str(int(i)) for i in list] 
    num = int("".join(lis)) 
    return(int(str(num),2))

# Convert the network's outputs into binary numbers
preds_conv=[]
for i in range(len(preds)):
    preds_conv.append(convert(preds[i]))
y_test1=[]
for i in range(len(y_test)):
    y_test1.append(convert(y_test[i]))

# Find of network by finding the number of correct predictions by the neural network#
# over the total number of predictions
y_pred = model.predict(X_test)
correct=[]
wrong=[]
for i in range(len(preds)):
    for j in range(15):
        if preds[i][j]==y_test[i][j]:
            correct.append(1)
        else:
            wrong.append(1)
# Print the error
print(len(correct)/(len(preds)*len(preds[0])))            

# Defines the test data and uscales w0
p_p=X_test[:,1]
w0_s=[]
for i in range(len(X_test[:,0])):
    w0_s.append(X_test[:,0][i]*(600.25-500)+500)
w0_s=np.array(w0_s)

# Initialise lists for different phases (converted numbers)
laserw=[]
laserp=[]
laserz=[]
randw=[]
randp=[]
randz=[]
rand1w=[]
rand1p=[]
rand1z=[]
rand2w=[]
rand2p=[]
rand2z=[]
rand3w=[]
rand3p=[]
rand3z=[]
rand4w=[]
rand4p=[]
rand4z=[]
rand5w=[]
rand5p=[]
rand5z=[]
rand6w=[]
rand6p=[]
rand6z=[]
rand7w=[]
rand7p=[]
rand7z=[]
rand8w=[]
rand8p=[]
rand8z=[]
becw=[]
becp=[]
becz=[]
otherw=[]
otherp=[]
otherz=[]

# Populate lists with points corresponding to certain phases (converted numbers)
for i in range(len(preds_conv)):
    if preds_conv[i] in [16384]:
        becw.append(w0_s[i])
        becp.append(p_p[i])
        becz.append(0)
    if preds_conv[i]==16768:
        rand1w.append(w0_s[i])
        rand1p.append(p_p[i])
        rand1z.append(3)
    if preds_conv[i]==31744:
        randw.append(w0_s[i])
        randp.append(p_p[i])
        randz.append(2)
    if preds_conv[i]==28672:
        rand2w.append(w0_s[i])
        rand2p.append(p_p[i])
        rand2z.append(4)
    if preds_conv[i]==15864:
        rand3w.append(w0_s[i])
        rand3p.append(p_p[i])
        rand3z.append(3)
    if preds_conv[i]==3968:
        rand4w.append(w0_s[i])
        rand4p.append(p_p[i])
        rand4z.append(6)
    if preds_conv[i]==19456:
        rand5w.append(w0_s[i])
        rand5p.append(p_p[i])
        rand5z.append(5)
    if preds_conv[i]==29696:
        rand6w.append(w0_s[i])
        rand6p.append(p_p[i])
        rand6z.append(8)
    if preds_conv[i]==12160:
        rand7w.append(w0_s[i])
        rand7p.append(p_p[i])
        rand7z.append(8)
    if preds_conv[i]==20480:
        rand8w.append(w0_s[i])
        rand8p.append(p_p[i])
        rand8z.append(5)
    if preds_conv[i]==0:
        laserw.append(w0_s[i])
        laserp.append(p_p[i])
        laserz.append(6)
    else:
        otherw.append(w0_s[i])
        otherp.append(p_p[i])
        otherz.append(1)

# Plot these to form a phase diagram
plt.plot([1/laserw[i] for i in range(len(laserw))],laserp,'o',color='b')
plt.plot([1/otherw[i] for i in range(len(otherw))],otherp,'s',color='g',markersize=10)
plt.plot([1/becw[i] for i in range(len(becw))],becp,'s',color='r',markersize=10)
plt.plot([1/randw[i] for i in range(len(randw))],randp,'s',color='k',markersize=10)
plt.plot([1/rand1w[i] for i in range(len(rand1w))],rand1p,'s',color='y',markersize=10)
plt.plot([1/rand2w[i] for i in range(len(rand2w))],rand2p,'s',color='c',markersize=10)
plt.plot([1/rand3w[i] for i in range(len(rand3w))],rand3p,'o',color='#FF1493')
plt.plot([1/rand4w[i] for i in range(len(rand4w))],rand4p,'o',color='#FF1491')
plt.plot([1/rand5w[i] for i in range(len(rand5w))],rand5p,'s',color='#FF1593',markersize=10)
plt.plot([1/rand7w[i] for i in range(len(rand7w))],rand7p,'o',color='#FF9495')
plt.yscale("log")

#Labelling the phases
plt.text(1.69e-3,0.001,'A',fontsize=15)
plt.text(1.78e-3,0.00001,'B',color='w',fontsize=15)
plt.text(1.82e-3,0.000001,'C',color='g',fontsize=15)
plt.ylabel(r"Pump rate $\Gamma{\uparrow}$",fontsize=15)
plt.text(1.87e-3,0.001,'D',fontsize=15)
plt.text(1.908e-3,0.01,'A',fontsize=15)
plt.xlabel(r"Thermalisation $\frac{1}{\lambda_{0}}$",fontsize=15)
plt.title("Discrete phase diagram for NN output",fontsize=15)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.tick_params(labelsize=15)