#Pipeline: StandardScaler for normalizing the values and doing the linear regression
baseline = make_pipeline(StandardScaler(with_mean=True), LinearRegression())
baseline.fit(X_tr, y_tr)  #Training the model with X,Y training
y_hat = baseline.predict(X_va)  #y_hat is the prevision for the validation set

#Doing the mean squared error between the prevision (y_hat) and the real values(y_va)
rmse = mean_squared_error(y_va, y_hat) 
rmse_baseline = np.sqrt(mean_squared_error(y_va, y_hat))
print("Baseline RMSE (validation split):", rmse_baseline)

#Plot y_^y
def plot_y_yhat(y_test,y_pred, plot_title = "plot"):
    labels = ['x_1','y_1','x_2','y_2','x_3','y_3']
    save_dir = "../outputs/pdf"
    os.makedirs(save_dir, exist_ok=True)
    MAX = 500 #Maximum number of points to plot for clarity
    if len(y_test) > MAX: 
        idx = np.random.choice(len(y_test),MAX, replace=False) #Randomly choose MAX indices to plot
    else:
        idx = np.arange(len(y_test)) #Use all indices if less than MAX
    plt.figure(figsize=(10,10))
    for i in range(6): #For each of the 6 target variables
        x0 = np.min(y_test[idx,i])
        x1 = np.max(y_test[idx,i])
        plt.subplot(3,2,i+1)
        plt.scatter(y_test[idx,i],y_pred[idx,i])
        plt.xlabel('True '+labels[i])
        plt.ylabel('Predicted '+labels[i])
        plt.plot([x0,x1],[x0,x1],color='red')
        plt.axis('square')
    save_path = os.path.join(save_dir, plot_title + '.pdf')
    plt.savefig(save_path)
    plt.show()
    

plot_y_yhat(y_va, y_hat, plot_title="baseline_validation")

#Preparing the dataset for training
full = train[~train["is_padding"]].copy()
X_full = full[FEATURES].values
y_full = full[TARGETS].values

#Baseline model
baseline_full = make_pipeline(StandardScaler(with_mean=True), LinearRegression())
baseline_full.fit(X_full, y_full)

#Preparing test
X_test = test[["t","x0_1","y0_1","x0_2","y0_2","x0_3","y0_3"]].values
pred = baseline_full.predict(X_test)

#Construct the submission
submission = pd.DataFrame({
    "Id": test["Id"].astype(np.int64),
    "x_1": pred[:,0], "y_1": pred[:,1],
    "x_2": pred[:,2], "y_2": pred[:,3],
    "x_3": pred[:,4], "y_3": pred[:,5],
})
submission.to_csv("../outputs/csv/baseline-model.csv", index=False)
print("Saved submission file: baseline-model.csv") 