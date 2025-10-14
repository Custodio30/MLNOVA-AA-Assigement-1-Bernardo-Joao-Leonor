##### Data visualization #####

#Creates a unique identifier for each trajectory
#Essential to group lines by trajectory (to exclude collisions and make correct division)
def add_traj_id(df, steps_per_traj= 257):
    df=df.copy() # Avoid modifying the original DataFrame
    df["traj_id"] = (df["Id"] // steps_per_traj) #Each trajectory has 257 steps, e.g index 0-256 -> traj_id 0, index 257-514 -> traj_id 1, etc.
    return df

#Identifies all the lines that have a colision with True
#Essential for filtering colisions and the trajectories with colisions 
def mark_padding(df):
    df=df.copy()
    feat_cols = [
        'x_1','y_1','v_x_1','v_y_1',
        'x_2','y_2','v_x_2','v_y_2',
        'x_3','y_3','v_x_3','v_y_3'
    ]
    df["is_padding"] = (df[feat_cols].abs().sum(axis=1)==0) #If all feature columns are zero, mark as colision (True)
    return df

#Identifies the initial position of each trajectorie and puts with x0_* and y0_*
#Essential to use the starting positions as static features.
def add_initial_positions(df):
    df=df.copy()
    first =df.loc[df.groupby("traj_id")["t"].idxmin(),  # Get the first row (t=0) for each trajectory
        ["traj_id","x_1","y_1","x_2","y_2","x_3","y_3"]].rename(columns={
        "x_1":"x0_1","y_1":"y0_1",
        "x_2":"x0_2","y_2":"y0_2",
        "x_3":"x0_3","y_3":"y0_3"
    })
    #Get the positions x,y for t=0 and rename the columns to x0_*, y0_* for each object
    df= df.merge(first, on="traj_id", how="left")
    return df

#Pipeline
def preprocess_train(df, steps_per_traj=257):
    df = add_traj_id(df, steps_per_traj=steps_per_traj)
    df = mark_padding(df)
    df = add_initial_positions(df)
    return df #Final dataset with traj_id, is padding, x0_*, y0_*, x_*, y_*, v_x_*, v_y_*

#Dataset visualization
#Important for exploring and documenting the database
def dataset_overview(df):
    n_rows, n_cols = df.shape #Number of lines and columns
    n_traj = df["traj_id"].nunique() #Number of unique trajectories
    steps_per_traj = df.groupby("traj_id")["t"].size() #Number of steps per trajectory
    colision_rows = 100 * df["is_padding"].mean()  #Percentage of padding rows

    info = pd.DataFrame({
        "Metric": [
            "Rows", 
            "Cols",
            "Number of trajectories with colisions",
            "Steps per trajectory (min)",
            "Steps per trajectory (max)",
            "Steps per trajectory (mean)",
            "% Colision rows rows",
            "Range of time [min,max]"
        ],
        "Value": [
            n_rows, 
            n_cols,
            n_traj,
            steps_per_traj.min(),
            steps_per_traj.max(),
            f"{steps_per_traj.mean():.2f}",
            f"{colision_rows:.2f}%",
            f"[{df['t'].min()}, {df['t'].max()}]"
        ]
    })
    return info

#Finds the trajectories that were classified has True in padding
#Important for insight into dynamics and collisions.
def find_collision_traj(df):
    has_pad = df.groupby("traj_id")["is_padding"].any() #Check which trajectories have any padding (True)
    colision_id = has_pad[has_pad].index #Get the traj_id of those trajectories
    return int(np.random.choice(colision_id)) if len(colision_id) > 0 else None #Choses one padding trajectory randomly 

#Histogram of non_paddings trajectories
#Important for understanding the effective trajectory lengths 
def plot_hist_nonpadding_lengths(df): 
    g = df.groupby("traj_id") #Join all the lines for the same trajectory
    lengths = (g["is_padding"].apply(lambda s: (~s).sum())).values #Count the number of non-padding lines per trajectory
    plt.figure(figsize=(6,4))
    plt.hist(lengths, bins=30)
    plt.xlabel("Non-padding length per trajectory")
    plt.ylabel("Count")
    plt.title("Distribution of effective trajectory lengths")
    plt.show()
 
#Histogram os a random trajectory
#Important for visualizing the movement patterns of objects in a trajectory
def plot_one_trajectory(df, traj_id=None, title_prefix="Trajectory"):
    if traj_id is None:
        traj_id = np.random.choice(df["traj_id"].unique()) #Choose a random trajectory if none is provided

    d = df.loc[df["traj_id"] == traj_id].sort_values("t") #Get all lines for the selected trajectory and sort by time
    plt.figure(figsize=(6,6))
    plt.plot(d["x_1"], d["y_1"], label="Obj 1")
    plt.plot(d["x_2"], d["y_2"], label="Obj 2")
    plt.plot(d["x_3"], d["y_3"], label="Obj 3")
    plt.plot(d["x_1"].iloc[0], d["y_1"].iloc[0], 's')
    plt.plot(d["x_2"].iloc[0], d["y_2"].iloc[0], 's')
    plt.plot(d["x_3"].iloc[0], d["y_3"].iloc[0], 's')
    plt.axis('equal')
    plt.legend()
    plt.title(f"{title_prefix} {traj_id}")
    plt.show()

#Plots 
def plot_t_hist(df):
    plt.figure(figsize=(6,4))
    plt.hist(df["t"], bins=50)
    plt.xlabel("t")
    plt.ylabel("count")
    plt.title("Histogram of timestep t")
    plt.show()

#Show the results
train = preprocess_train(data, steps_per_traj=257)
overview = dataset_overview(train)
display(overview)
plot_one_trajectory(train, traj_id=None, title_prefix="Normal trajectory") #Plot a random trajectory
col_traj = find_collision_traj(train)
if col_traj is not None:
    plot_one_trajectory(train, traj_id=col_traj, title_prefix="Collision trajectory") #Plot a random collision trajectory
else:
    print("No collision trajectories found (no padding)")

plot_t_hist(train)
plot_hist_nonpadding_lengths(train)

##### Data splitting #####

#Definition of features and targets
FEATURES = ["t","x0_1","y0_1","x0_2","y0_2","x0_3","y0_3"]
TARGETS  = ["x_1","y_1","x_2","y_2","x_3","y_3"]

#Defining split function in 70% train, 15% test and 15%validation
def make_train_val_test_split( 
    dataset: pd.DataFrame,
    features: list = FEATURES,
    targets: list  = TARGETS,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42 #For reproducibility of the random split
):
    #Making sure that the sum of the fractions is 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-8, "Sum of fractions is 1"
    
    #Removing all the lines with padding 
    traj_with_padding = dataset.loc[dataset["is_padding"], "traj_id"].unique()
    
    #Removing colisions
    collision_mask = (dataset["t"] == 0) & \
                     (dataset[["x_1","y_1","x_2","y_2","x_3","y_3"]] == 0).all(axis=1)
    
    traj_with_collision = dataset.loc[collision_mask, "traj_id"].unique()
    
    lines_with_pad_or_collision = dataset["traj_id"].isin(traj_with_padding) | dataset["traj_id"].isin(traj_with_collision)

    print("Total of lines with padding and colisions:", lines_with_pad_or_collision.sum())
    print("Total of trajectories with padding and colisions:", dataset.loc[lines_with_pad_or_collision, 'traj_id'].nunique())
    
    dataset= dataset[~dataset["traj_id"].isin(traj_with_padding) & 
                        ~dataset["traj_id"].isin(traj_with_collision)]
    
    print("Total of lines without padding and colisions:", dataset.shape[0])
    print("Total of trajectories without padding and colisions:", dataset['traj_id'].nunique())

    #Making sure that the trajectories are unique 
    unique_traj = np.array(dataset["traj_id"].unique()) #Get unique trajectory IDs
    np.random.seed(random_state) #Set random seed for reproducibility
    np.random.shuffle(unique_traj) #Shuffle the trajectory IDs randomly

    #Total number of unique trajectories and dividing fot each split according to the fractions and the total number of unique trajectories
    n_total = len(unique_traj) 
    n_train = int(train_size * n_total)
    n_val   = int(val_size * n_total)
    n_test = n_total - n_train - n_val

    #Getting the trajectory IDs for each split in an indenpendent manner
    tr_ids = unique_traj[:n_train]
    va_ids = unique_traj[n_train:n_train+n_val]
    te_ids = unique_traj[n_train+n_val:]
    
    #Creating a dataset with the trajectory IDs per split
    tr = dataset[dataset["traj_id"].isin(tr_ids)].copy() 
    va = dataset[dataset["traj_id"].isin(va_ids)].copy()
    te = dataset[dataset["traj_id"].isin(te_ids)].copy()
    
    #Extracting X and Y arrays for each ID of each split 
    X_tr, y_tr = tr[features].values, tr[targets].values
    X_va, y_va = va[features].values, va[targets].values
    X_te, y_te = te[features].values, te[targets].values

    return X_tr, y_tr, X_va, y_va, X_te, y_te, tr, va, te

#Executing split
X_tr, y_tr, X_va, y_va, X_te, y_te, tr_df, va_df, te_df = make_train_val_test_split(train)

print("Train:", X_tr.shape, y_tr.shape)
print("Val:  ", X_va.shape, y_va.shape)
print("Test: ", X_te.shape, y_te.shape)