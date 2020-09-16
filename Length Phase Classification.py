#Import Packages
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from sklearn.preprocessing import MinMaxScaler

#Import Data (v2)
order = 3 # this is the order of approximation
modes = 15 # number of modes
mode_val=[]
pump_poww=[]

# Load data for each cut-off wavelength w0
for w0 in np.arange(570.0,600.5,0.25):
    alldata=np.load(r"data_file_path".format(w0,modes,order),encoding='bytes',allow_pickle=True).item(0)
    mode_val.append(list(alldata.values())[1])
    pump_poww.append(list(alldata.values())[0])
training_w0=[]
training_pump=[]

# Format the data to have the same number of elements in each parameter array
for i in range(122):
    if len(mode_val[i]) == 129:
        for j in range(129):
            training_w0.append(np.arange(570.0,600.5,0.25)[i])
            training_pump.append(pump_poww[i][j])

# Define the threshold to classify condensed modes; change as needed
threshold=1e4

# Find the number of condensed modes for each combinating of w0 and p_p
y=[]
for w0 in range(122):
    for pump_power in range(129):
        if len(mode_val[w0]) == 129:
            populated_modes=[]
            for mode in range(len(mode_val[w0][pump_power])):
                if mode_val[w0][pump_power][mode]>=threshold:
                    populated_modes.append(mode_val[w0][pump_power][mode])
            y.append(len(populated_modes))

#Scaling the lengths to be between 0 and 1
for i in range(len(y)):
    y[i]=(y[i]-min(y))/(max(y)-min(y))

#Formatting Data to pandas dataframes for keras handling
x=[]
x.append(training_w0)
x.append(training_pump)
x=pd.DataFrame(data=x)
X=x.T
X=pd.DataFrame.to_numpy(X)

#Splitting the Data into test/train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03, random_state=42)

#Plotting the test data
plt.figure()
plt.plot(y_test, color = 'red', label = 'Real data')
plt.title('Test data')
plt.show()

#Normalising w0 and p_p
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(y_train)
y_train=np.array(y_train)

# Initialising the network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2,)),
    keras.layers.Dense(2, input_dim=2, activation='relu'),
	keras.layers.Dense(26,  activation='relu', kernel_initializer='uniform'),
    keras.layers.Dense(26,  activation='relu', kernel_initializer='uniform'),
    keras.layers.Dense(1,activation='sigmoid'),
])

# Compile the network model; use adam optimiser and mse cost function
model.compile(loss='mse', optimizer='adam',metrics=['mse'])

# Train the network using training data; 10 epochs, batch size of 100
model.fit(X_train, y_train, epochs=10, batch_size=100)

#Testing the neural network on test data
y_pred=model.predict(X_test)

#Denormalising the neural network test data and rounding it to the nearest integer
y_pred = [round(y_pred[i][0]*13) for i in range(len(X_test))]
y_test = [round(y_test[i]*13) for i in range(len(X_test))]

#Plotting the data used to test the neural network, with that the neural network predicts.
plt.figure()
plt.plot(y_pred,color='k')
plt.plot(y_test,color='r')
plt.title('Comparing Error in prediction by neural network')
plt.legend(('Data predicted by Neural Network','Test Data'))
plt.grid()
plt.show()

#Plotting a histogram of the instances of each length classification of the test data
# to show bias
plt.figure()
plt.hist(y_test,bins=100,align='mid')
plt.xlabel("Number of condensed modes:")
plt.ylabel('Number of instances:')
plt.grid()
plt.show()

# Define test pump power and cut-off wavelength data
p_p=X_test[:,1]
w0_s=X_test[:,0]
yy=y_test
z=np.array(yy)
y = p_p
x = np.array([w0_s[i]*(600.25-500)+500 for i in range(len(w0_s))]) 

# Interpolate to form a phase diagram from the 'true' test data phases
z=np.array(z)
nInterp = 200
xi, yi = np.linspace(x.min(), x.max(), nInterp), np.linspace(y.min(), y.max(), nInterp)
xi, yi = np.meshgrid(xi, yi)
zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='linear')
plt.figure()
plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
       extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto') 

# Labelling plot
plt.xlabel(r"$\omega_{0}$")
plt.ylabel(r"$\Gamma_{\uparrow}$",rotation=0)
plt.title(r"Variation of output error with $\Gamma_{\uparrow}$ and $\omega_{0}$")
cb=plt.colorbar()
cb.set_label("# condensed modes")
plt.show()

# Repeat for predicted phase diagram from network's outputs
y=y_pred
z=np.array(y)
y = p_p
x = np.array([1/(w0_s[i]*(600.25-500)+500) for i in range(len(w0_s))]) 
z=np.array([int(z[i]) for i in range(len(z))])
nInterp = 200
xi, yi = np.linspace(x.min(), x.max(), nInterp), np.linspace(y.min(), y.max(), nInterp)
xi, yi = np.meshgrid(xi, yi)
zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='linear')
plt.figure()
plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
       extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto') 

# Labelling plot
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$\Gamma_{\uparrow}$",rotation=0)
plt.yscale("log")
plt.title(r"Variation of $N^{o}$ condensed modes with $\Gamma_{\uparrow}$ and $\lambda_{0}$")
cb=plt.colorbar()
cb.set_label(r"$N^{o}$ condensed modes")
plt.show()

# Repeat for the difference between true phase output and that predicted by network
yp=y_pred
yy=[]
for i in range(len(y)):
    yy.append((abs(yp[i]-y_test[i]))/15)
z=np.array(yy)
y = p_p
x = np.array([1/(w0_s[i]*(600.25-500)+500) for i in range(len(w0_s))]) 
nInterp = 200
xi, yi = np.linspace(x.min(), x.max(), nInterp), np.linspace(y.min(), y.max(), nInterp)
xi, yi = np.meshgrid(xi, yi)
zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='linear')
plt.figure()
plt.imshow(zi, vmin=z.min(), vmax=.4, origin='lower',
       extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto') 

# Labelling plot
plt.xlabel(r"$\gamma$:")
plt.ylabel(r"$\Gamma_{\uparrow}$:",rotation=0)
plt.title(r"Variation of output error with $\Gamma_{\uparrow}$ and $\gamma$")
cb=plt.colorbar()
cb.set_label("Error")
plt.show()

# Create a plot to show how the mode populations vary with pump power
n = 121  # w0 index
x = pump_poww[0]

# Initialise and populate lists for different modes
y1 = []
y2 = []
y3 = []
y13 = []
for i in range(129):
    y1.append(mode_val[n][i][0])
for i in range(129):
    y2.append(mode_val[n][i][1])
for i in range(129):
    y3.append(mode_val[n][i][2])
for i in range(129):
    y13.append(mode_val[n][i][12])
for i in range(129):
x.sort()
y1.sort()
y2.sort()
y3.sort()
y13.sort()
plt.xscale('log')
plt.yscale('log')
x.sort()
y1.sort()
y2.sort()
y3.sort()
thresh=[10**4 for i in range(len(x))]
plt.plot(x,thresh,'--',label="Threshold")
plt.plot(x,y1,label=r"$1^{st}$ mode")
plt.plot(x,y2,label=r"$2^{nd}$ mode")
plt.plot(x,y3,label=r"$3^{rd}$ mode")
plt.plot(x,y13,label=r"$13^{th}$ mode")
plt.legend(fontsize=12)
plt.xlabel(r"$\Gamma_{\uparrow}$",fontsize=15)
plt.tick_params(labelsize=15)
plt.ylabel("Mode population",fontsize=15)
plt.xscale('log')
plt.yscale('log')