#Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Directories
data=pd.read_csv("~/Downloads/X_train.csv")
X=pd.read_csv("~/Downloads/X_test.csv")

####Documentation and exploration of the data
#Dataset size
print(data.head)
print(data.columns)

#Trajectories counting the 0s
trajectory=(len(data)-1)//258
print(f"Trajectories counting collisions: {trajectory}")

#Valid trajectories
idx=np.hstack((0,data[data.t==10].index.values+1))
idx.shape, data.t.min(), data.t.max() #We have 4054 valid trajectories each one going from t=0...t=10

#Visualize some trajectories
k=np.random.randint(idx.shape[0])
print(k) #Selects the trajectory number k

pltdix=range(idx[k],257+idx[k])
pltsquare=idx[k]
plt.plot(data.x_1[pltdix],data.y_1[pltdix])
plt.plot(data.x_2[pltdix],data.y_2[pltdix])
plt.plot(data.x_3[pltdix],data.y_3[pltdix])

plt.plot(data.x_1[pltsquare],data.y_1[pltsquare], "s")
plt.plot(data.x_2[pltsquare],data.y_2[pltsquare], "s")
plt.plot(data.x_3[pltsquare],data.y_3[pltsquare], "s") #Plots this random trajectory k

#Cheking missing values
print("Missing values per column:")
print(data.isnull().sum())

#Statistics
print("Data statistics:")
print(data.describe()) 

#Cheking how many timestpes with colisions (when t=0, and x1,y1,x2,y2,x3,y3=0)
collision = (data[['x_1','y_1','x_2','y_2','x_3','y_3']] == 0).all(axis=1)
print(f"Timestpes with colisions: {collision.sum()}")


###Data splitting and pipeline construction
#Each trajectory for each object with the same values of x_0_y_0 has the same value init_keys
#Makes the data indpendent 

init_keys = (
    X[["x0_1","y0_1","x0_2","y0_2","x0_3","y0_3"]]
    .astype(str)
    .agg("_".join, axis=1)
)

#Reusable split function
def split_train_val_test(X, Y, init_keys, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    
    unique_keys = np.unique(init_keys)
    np.random.seed(random_state)
    np.random.shuffle(unique_keys)
    
    n_total=len(unique_keys)
    n_train=int(train_size * n_total)
    n_val=int(val_size * n_total)
    
    train_keys=unique_keys[:n_train]
    val_keys=unique_keys[n_train:n_train+n_val]
    test_keys=unique_keys[n_train+n_val:]
    
    train_mask=np.isin(init_keys, train_keys)
    val_mask=np.isin(init_keys, val_keys)
    test_mask=np.isin(init_keys, test_keys)
    
    return (X[train_mask], Y[train_mask],
            X[val_mask], Y[val_mask],
            X[test_mask], Y[test_mask])

Y= data[["x_1","y_1","x_2","y_2","x_3","y_3"]].values

#Calling the function
X_train, Y_train, X_val, Y_val, X_test, Y_test = split_train_val_test(X, Y, init_keys, train_size=0.2, val_size=0.05, test_size=0.05)

print("Train:", X_train.shape, Y_train.shape)
print("Val:", X_val.shape, Y_val.shape)
print("Test:", X_test.shape, Y_test.shape)


