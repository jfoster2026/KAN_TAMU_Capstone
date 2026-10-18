import matplotlib
matplotlib.use("Agg")  # ADDED FOR GUI: figures are drawn into the GUI instead of separate windows
import kan, torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from kan import ex_round

gui_state = {}  # ADDED FOR GUI: stores last trained model + ranges so single_test() can be triggered

def normalize(x):  # min-max scaling
    if x.min() == x.max():
        return (x*0 + 1)
    return (x - x.min()) / (x.max() - x.min())

def denormalize(x, x_min, x_max):  # min-max unscaling
    return x * (x_max - x_min) + x_min

def train_model(expmnt, kan_width, kan_grid, kan_k, all_time, all_PCD, all_volt, all_bias, all_load, 
                time_mx_mn, PCD_mx_mn, volt_mx_mn, bias_mx_mn, load_mx_mn,
                train_losses, test_losses, train_ratio, LOO_test = False):

    # kan_width must be a list
    print(f"Training on {train_ratio*100}% of data")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device)

    # initialize KAN
    model = kan.KAN(width=kan_width, grid=kan_grid, k=kan_k, seed=1, grid_range= [0,1], device=str(device))

    # declare variables used elsewhere in program.
    expmnt_size = []

    for exp_num in expmnt:
        if exp_num == 0: continue
        # collect data from dataset
        file_name = f"TestData\\LFXR_Lovejoy_Sept2021_{exp_num}.csv"
        df_train = pd.read_csv(file_name, usecols=[0,1,3])

        df_start_stop = pd.read_csv("start_stop_pts.csv")
        expmnt_row = exp_num - 27271   # the first experiment is on row index 1
        volt_start = int(df_start_stop['volt_start'].iloc[expmnt_row])
        volt_stop = int(df_start_stop['volt_stop'].iloc[expmnt_row])
        rad_start = int(df_start_stop['rad_start'].iloc[expmnt_row])
        rad_stop = int(df_start_stop['rad_stop'].iloc[expmnt_row])

        expmnt_size.append(volt_stop - volt_start)

        bias = float(df_start_stop['bias'].iloc[expmnt_row])
        load = float(df_start_stop['load'].iloc[expmnt_row])

        
        #print(volt_start, volt_stop, rad_start, rad_stop)

        time = df_train['time1'].iloc[volt_start:volt_stop].to_numpy(dtype = np.float32)
        # Make time in microseconds
        time = time * 1e6
        time = time - time.min() # shift time to start at 0


        PCD = df_train['PCD3_B'].iloc[rad_start:rad_stop].to_numpy(dtype = np.float32)

        # output voltage
        volt = df_train['Diode'].iloc[volt_start:volt_stop].to_numpy(dtype = np.float32) #.reshape(-1,1)
        #volt = volt / (bias if bias > 1 else 1)
    
        all_time = np.append(all_time, time)
        all_PCD = np.append(all_PCD, PCD)
        all_volt = np.append(all_volt, volt)
        all_bias = np.append(all_bias, np.full_like(time, bias))
        all_load = np.append(all_load, np.full_like(time, np.log10(load))) # use logarithmic scale for load to prevent NANs.


    # time_mx_mn[0] = all_time.max()
    # time_mx_mn[1] = all_time.min() 
    # all_time = normalize(all_time)  # normalize time to [0,1]
    all_time = (all_time - time_mx_mn[1]) / (time_mx_mn[0] - time_mx_mn[1])

    # volt_mx_mn[0] = all_volt.max()
    # volt_mx_mn[1] = all_volt.min()
    # all_volt = normalize(all_volt)  # normalize voltage to [0,1]
    all_volt = (all_volt - volt_mx_mn[1]) / (volt_mx_mn[0] - volt_mx_mn[1])

    PCD_mx_mn[0] = all_PCD.max()
    PCD_mx_mn[1] = all_PCD.min()  
    #all_PCD = normalize(all_PCD)  # normalize PCD to [0,1]
    all_PCD = (all_PCD - PCD_mx_mn[1]) / (PCD_mx_mn[0] - PCD_mx_mn[1])

    all_bias = (all_bias - bias_mx_mn[1]) / (bias_mx_mn[0] - bias_mx_mn[1])
    all_load = (all_load - load_mx_mn[1]) / (load_mx_mn[0] - load_mx_mn[1]) 

    # Make input matrix
    X = np.column_stack((all_time, all_PCD, all_bias, all_load))  # shape (num_samples, 4)


    #pull training points from the data
    np.random.seed(None)
    # need to get certain number of points from each experiment, so we don't just pull the first 70% of the data
    slice_idx = 0
    test_indices = np.array([], dtype=int)
    
    for idx, i in enumerate(expmnt):
        if i == 0: continue
        slice_idx += expmnt_size[idx]
        exp_indices = np.arange(slice_idx - expmnt_size[idx], slice_idx)
        exp_test_indices = np.random.choice(exp_indices, 
                            round(expmnt_size[idx] * (1 - train_ratio)), replace=False)
        test_indices = np.concatenate((test_indices, exp_test_indices))

    X_test = X[test_indices]
    n_volt_test = all_volt[test_indices] # Correct Answers

    #pull the rest for training
    train_indices = np.setdiff1d(np.arange(X.shape[0]), test_indices)
    X_train = X[train_indices]
    n_volt_train = all_volt[train_indices]

    dataset = {
        "train_input": torch.from_numpy(X_train).to(device),
        "train_label": torch.from_numpy(n_volt_train).reshape(-1,1).to(device),
        "test_input":  torch.from_numpy(X_test).to(device),
        "test_label":  torch.from_numpy(n_volt_test).reshape(-1,1).to(device),
    }

    print(X.shape, all_volt.shape)
    
    for grid in [0]:
        if grid != 0:
            print(f"Refined to {kan_grid + grid}")
            model.refine(kan_grid + grid)
        results = model.fit(dataset,
                opt="Adam", # Adam (faster) or LBFGS (slower)
                steps=500,
                lr=0.03, # was 0.016
                update_grid=False,  # set to false for single experiment training.
                start_grid_update_step= 100,
                stop_grid_update_step= 120,
                lamb=0.1,        # master scale — tune this first
                lamb_l1=0.0,      # high — you want to prune weak connections
                lamb_entropy=0.0, # high — encourages interpretable simple functions
                lamb_coef=0.0,    # usually redundant if l1 is active
                lamb_coefdiff=0.00) # smooth splines — important for noisy data)
    train_losses += results['train_loss']
    test_losses += results['test_loss']
    # The losses are already NRMSE because all of the data (input and output) is normalized to the range of 0 to 1

    # # plot the two losses
    # plt.plot(train_losses)
    # plt.plot(test_losses)
    # plt.legend(['train', 'test'])
    # plt.ylabel('RMSE')
    # plt.xlabel('step')
    # plt.yscale('log')
    # plt.show()

    return model, dataset, expmnt_size





def plot_results(model, dataset, time_mx_mn, PCD_mx_mn, volt_mx_mn, expmnt, expmnt_size, train_ratio):    

    t = dataset["test_input"][:, 0]   # assumes column 0 is time
    r = dataset["test_input"][:, 2]   # assumes column 2 is radiation
    v = dataset["test_label"]         # voltage target (expected value, correct answer)
    b = dataset["test_input"][:, 3]   # assumes column 3 is bias

    with torch.no_grad():
        v_pred = model(dataset["test_input"])
    # MSE and RMSE in percentage form
    mse = torch.mean((v_pred - v) ** 2).item() 
    print(f"{expmnt} Test MSE: {mse * 100:.3f}%")
    rmse = mse ** 0.5
    nrmse = rmse / float((v.max() - v.min())) # this gives the NRMSE relative to the values in the experiment set.
    print(f"{expmnt} Test NRMSE: {nrmse * 100:.3f}%") 
    

    test_results_idx = 0
    # test_indice are already defined
    for idx, i in enumerate(expmnt):
        if i == 0: continue
        # Get the number of test points for this experiment
        num_test_pts = round(expmnt_size[idx] * (1 - train_ratio))

        # Extract the relevant portion of the data for this experiment
        t_np = t.detach().cpu().numpy()[test_results_idx:test_results_idx + num_test_pts]
        r_np = r.detach().cpu().numpy()[test_results_idx:test_results_idx + num_test_pts]
        v_np = v.detach().cpu().numpy()[test_results_idx:test_results_idx + num_test_pts]
        v_pred_np = v_pred.detach().cpu().numpy()[test_results_idx:test_results_idx + num_test_pts]
        bias = b[test_results_idx]

        # Denormalize the data for plotting ; xxxx_mx_min[0] is max, xxxx_mx_mn[1] is min
        t_np = denormalize(t_np, time_mx_mn[1], time_mx_mn[0])  # unnormalize time
        r_np = denormalize(r_np, PCD_mx_mn[1], PCD_mx_mn[0])  # unnormalize PCD
        v_np = denormalize(v_np, volt_mx_mn[1], volt_mx_mn[0])  #* float(bias if bias > 1 else 1) # unnormalize voltage
        v_pred_np = denormalize(v_pred_np, volt_mx_mn[1], volt_mx_mn[0]) #* float(bias if bias > 1 else 1) # unnormalize voltage

        test_results_idx += num_test_pts

        # NOT BEING USED AT THE MOMENT
        # arr = np.column_stack((t_np / 1e6, r_np)) # make time seconds again
        # sorted_arr = arr[np.argsort(arr[:, 0])]
        # # Define folder and file path
        # folder = "pulse_text_files"
        # os.makedirs(folder, exist_ok=True)  # Creates folder if it doesn't exist
        # file_path = os.path.join(folder, f"pulse_{i}.txt")

        # # Save array to .txt file
        # np.savetxt(
        #     file_path,
        #     sorted_arr,
        #     fmt="%.9e",          # Integer format (use "%.4f" for floats)
        #     delimiter=" ",     # Whitespace delimiter
        #     )

        plt.figure()
        plt.scatter(t_np, v_np, s=20, label="voltage (true)")
        plt.scatter(t_np, v_pred_np, s=7, label="voltage (pred)")
        #plt.scatter(t_np, r_np, s=7, label="radiation")
        plt.xlabel("time (us)")
        plt.ylabel("voltage (V)")
        plt.title(f"Experiment {i}")
        plt.legend()
        plt.show()
    return round(nrmse * 100, 2) # return the NRMSE as a percent



# Used to test the model using a single experiment to test generalization.
def plot_losses(filename_list, arch):
    # plot the losses from perform_training() as a scatter plot
    # arch is the architecture of the model
    # can receive multiple files names to compare those two runs
    
    dot_colors = ['g','r','b','o']
    special_lables = ["New", "Old"]
    plt.figure(figsize=(12,4))
    for idx, filename in enumerate(filename_list):
        losses_df = pd.read_csv(filename)
        plt.plot(range(len(losses_df["Error"])), losses_df["Error"], 
                    marker = 'o', linestyle = '-', color = dot_colors[idx], label = special_lables[idx]) #arch[idx]) 
        plt.xlabel("Trial")
        plt.ylabel("Error (%)")
        plt.title("Error for each trial")
        #plt.axhline(10, color = 'k', linestyle="--", linewidth = 1) # k is black
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(f"training_errors_{arch[0]}.png")
    plt.show()
    

    

def single_test(model, exp_num, time_mx_mn, PCD_mx_mn, volt_mx_mn, bias_mx_mn, load_mx_mn):
    print(f"Testing with experiment {exp_num}...")
    #reset variables
    all_time = np.array([], dtype = "float32")
    #time_mx_mn = [0.0, 0.0]
    all_PCD = np.array([], dtype = "float32")
    #PCD_mx_mn = [0.0, 0.0]
    all_volt = np.array([], dtype = "float32")
    #volt_mx_mn = [0.0, 0.0]
    all_bias = np.array([], dtype = "float32")
    all_load = np.array([], dtype = "float32")


    expmnt_size = []
    time_exp_len = []
    
    # collect data from dataset
    file_name = f"TestData\\LFXR_Lovejoy_Sept2021_{exp_num}.csv"
    df_train = pd.read_csv(file_name, usecols=[0,1,3])

    df_start_stop = pd.read_csv("start_stop_pts.csv")
    expmnt_row = exp_num - 27271   # the first experiment is on row index 1
    volt_start = int(df_start_stop['volt_start'].iloc[expmnt_row])
    volt_stop = int(df_start_stop['volt_stop'].iloc[expmnt_row])
    rad_start = int(df_start_stop['rad_start'].iloc[expmnt_row])
    rad_stop = int(df_start_stop['rad_stop'].iloc[expmnt_row])

    expmnt_size.append(volt_stop - volt_start)

    bias = float(df_start_stop['bias'].iloc[expmnt_row])
    load = float(df_start_stop['load'].iloc[expmnt_row])


    time = df_train['time1'].iloc[volt_start:volt_stop].to_numpy(dtype = np.float32)
    # Make time in microseconds
    time = time * 1e6
    time = time - time.min()


    PCD = df_train['PCD3_B'].iloc[rad_start:rad_stop].to_numpy(dtype = np.float32)

    # output voltage
    volt = df_train['Diode'].iloc[volt_start:volt_stop].to_numpy(dtype = np.float32) #.reshape(-1,1)
    #volt = volt / (bias if bias > 1 else 1)
    

    all_time = np.append(all_time, time)
    all_PCD = np.append(all_PCD, PCD)
    all_volt = np.append(all_volt, volt)
    all_bias = np.append(all_bias, np.full_like(time, bias))
    all_load = np.append(all_load, np.full_like(time, np.log10(load))) # use logarithmic scale for load to prevent NANs.
    
    
    #time_mx_mn[0] = all_time.max()
    #time_mx_mn[1] = all_time.min()
    time_exp_len.append(np.full_like(all_time, all_time.max()))    
    #all_time = normalize(all_time)  # normalize time to [0,1]
    all_time = (all_time - time_mx_mn[1]) / (time_mx_mn[0] - time_mx_mn[1])
    print(f"Max normalized time value: {all_time.max()}")

    #volt_mx_mn[0] = all_volt.max()
    #volt_mx_mn[1] = all_volt.min()
    #all_volt = normalize(all_volt)  # normalize voltage to [0,1]
    all_volt = (all_volt - volt_mx_mn[1]) / (volt_mx_mn[0] - volt_mx_mn[1])

    #PCD_mx_mn[0] = all_PCD.max()
    #PCD_mx_mn[1] = all_PCD.min()  
    #all_PCD = normalize(all_PCD)  # normalize PCD to [0,1]
    all_PCD = (all_PCD - PCD_mx_mn[1]) / (PCD_mx_mn[0] - PCD_mx_mn[1])

    all_bias = (all_bias - bias_mx_mn[1]) / (bias_mx_mn[0] - bias_mx_mn[1])
    all_load = (all_load - load_mx_mn[1]) / (load_mx_mn[0] - load_mx_mn[1])    


    # Make input matrix
    X = np.column_stack((all_time, all_PCD, all_bias, all_load))  # shape (num_samples, 4)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = {
        #"train_input": torch.from_numpy(X).to(device),
        #"train_label": torch.from_numpy(n_volt).to(device),
        "test_input":  torch.from_numpy(X).to(device),
        "test_label":  torch.from_numpy(all_volt).reshape(-1,1).to(device),
    }
    exp_num = [exp_num]
    nrmse = plot_results(model, dataset, time_mx_mn, PCD_mx_mn, volt_mx_mn, exp_num, expmnt_size, train_ratio = 0)
    return nrmse

    

def perform_training(kan_width, kan_grid, kan_k, LOO = False):

    # IF something does not work, the bias is being used to scale down voltage and then scale it up again in plotting.
    # add a  #  to a row to make the training stop early
    if LOO:
        filename = "LOO_training_subsets.csv"
        subsets_df = pd.read_csv(filename)
        train_these = subsets_df.drop(columns=["Name", "Test", "Reasoning"]).to_numpy().tolist()
        test_these = subsets_df["Test"]
        
    else:
        filename = "training_subsets.csv"
        subsets_df = pd.read_csv(filename)
        train_these = subsets_df.drop(columns=["Name", "Reasoning"]).to_numpy().tolist()


    final_train_losses = []
    final_test_losses = []
    new_file_row_count = 0
    for idx, i in enumerate(train_these):
        if '#' in subsets_df["Name"].iloc[idx]: 
            print(f"Terminated early at index {idx}")
            break
        print(f"Processing experiment {i}...")
        new_file_row_count += 1
        #reset variables
        all_time = np.array([], dtype = "float32")
        time_mx_mn = [6.5, 0.0]
        all_PCD = np.array([], dtype = "float32")
        PCD_mx_mn = [0.0, 0.0]
        all_volt = np.array([], dtype = "float32")
        volt_mx_mn = [15.0, -10.0]
        all_bias = np.array([], dtype = "float32")
        bias_mx_mn = [5.25, 0]
        all_load = np.array([], dtype = "float32")
        load_mx_mn = [6,1.6989]

        train_losses = []
        test_losses = []
        train_ratio = 0.7  # Use 70% of the data for training
        if any(item in i for item in [27277, 27292, 27293, 27306]): 
            print(f"This subset {i} has an unusable experiment. Skipping")
        else:
            expmnt = i


            M1, data1, expmnt_size = train_model(expmnt, kan_width, kan_grid, kan_k, all_time, all_PCD, all_volt, all_bias, 
                                                all_load, time_mx_mn, PCD_mx_mn, volt_mx_mn, bias_mx_mn, load_mx_mn,
                                                train_losses, test_losses, train_ratio, LOO_test = LOO)
            gui_state.update(model=M1, time_mx_mn=time_mx_mn, PCD_mx_mn=PCD_mx_mn, volt_mx_mn=volt_mx_mn,
                             bias_mx_mn=bias_mx_mn, load_mx_mn=load_mx_mn)  # ADDED FOR GUI
            final_train_losses.append(round(float(test_losses[-1])*100,3)) # already in percentage form and rounded
            nrmse = plot_results(M1, data1, time_mx_mn, PCD_mx_mn, volt_mx_mn, expmnt, expmnt_size, train_ratio)
            if LOO:
                final_test_losses.append(single_test(M1, subsets_df["Test"].iloc[idx],
                                                     time_mx_mn, PCD_mx_mn, volt_mx_mn, bias_mx_mn, load_mx_mn))

    train_loss_df = pd.DataFrame({"Name"  : subsets_df["Name"].iloc[:new_file_row_count], 
                                    "Error" : final_train_losses,
                                    "Reasoning": subsets_df["Reasoning"].iloc[:new_file_row_count]})
    train_loss_df.to_csv(f"final_train_loss_{kan_width[1][0]}.csv", index=False) # for some reason its making nested lists.
    print(f"Training Test Error Rates: {final_train_losses}")
    if LOO:
        print(f"LOO Testing Error Rates: {final_test_losses}")
        plt.scatter(range(len(final_test_losses)),final_test_losses)
        plt.title("LOO Test Errors")
        plt.xlabel("Trial")
        plt.ylabel("NRMSE in %")
        plt.show()



# perform_training([4,8,8,1], 30, 3)   # now launched from the GUI



def perform_LOO_training():
    subsets_df = pd.read_csv("training_subsets.csv")
    LOO_subsets_df = pd.DataFrame(columns=["Name", "Exp01", "Exp02", "Exp03", "Exp04", "Exp05", "Test", "Reasoning"])

    for i in range(subsets_df.shape[0]):
        row = subsets_df.iloc[i,:].to_list()
        if row[3] == 0: continue # make sure experiment has at least 3 experiments
        for j in row[1:-1]:
            if j == 0: continue
            train = row[1:-1]
            train.remove(j)
            new_row = [row[0]]
            new_row += train
            new_row += [int(j), row[-1]]
            LOO_subsets_df.loc[len(LOO_subsets_df)] = new_row
    
    LOO_subsets_df.to_csv("LOO_training_subsets.csv", index = False)

    perform_training([4,8,8,1], 6, 3, True)


 
#perform_LOO_training()

# plot_losses(["final_train_loss_8_new.csv","final_train_loss_8.csv"], [[5,8,1],[4,8,1]])   # now launched from the GUI


# =====================================================================
# ======================= GUI (added) =================================
# =====================================================================
import sys, re, ast, queue, threading, traceback
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk

class _QueueWriter:
    """Redirects print() output from the worker thread into the GUI log."""
    def __init__(self, q):
        self.q = q
    def write(self, s):
        if s:
            self.q.put(("log", s))
    def flush(self):
        pass


class KANApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("myKAN - Training / Single Test")
        self.geometry("1250x850")
        self.q = queue.Queue()
        self.busy = False
        self.latest_nrmse = None
        self._build()
        self.after(100, self._poll)

    # ---------------- layout ----------------
    def _build(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side="left", fill="y")

        # --- training controls ---
        tf = ttk.LabelFrame(left, text="Training (perform_training)", padding=8)
        tf.pack(fill="x", pady=4)
        self.width_var = tk.StringVar(value="[4,8,8,1]")
        self.grid_var = tk.StringVar(value="30")
        self.k_var = tk.StringVar(value="3")
        for r, (lbl, var) in enumerate([("kan layers", self.width_var),
                                        ("kan grid points", self.grid_var),
                                        ("kan spline order", self.k_var)]):
            ttk.Label(tf, text=lbl).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(tf, textvariable=var, width=14).grid(row=r, column=1, pady=2)
        self.train_btn = ttk.Button(tf, text="Run model training", command=self.on_train)
        self.train_btn.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        self.loo_btn = ttk.Button(tf, text="Run LOO training", command=self.on_loo)
        self.loo_btn.grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)

        # --- single test controls ---
        sf = ttk.LabelFrame(left, text="Single test (single_test)", padding=8)
        sf.pack(fill="x", pady=4)
        ttk.Label(sf, text="Experiment #").grid(row=0, column=0, sticky="w", pady=2)
        self.exp_var = tk.StringVar(value="27271")
        ttk.Entry(sf, textvariable=self.exp_var, width=14).grid(row=0, column=1, pady=2)
        self.single_btn = ttk.Button(sf, text="Run a single experiment", command=self.on_single)
        self.single_btn.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        ttk.Label(sf, text="Uses the most recently trained model.",
                  foreground="gray").grid(row=2, column=0, columnspan=2, sticky="w")

        # --- results ---
        rf = ttk.LabelFrame(left, text="NRMSE", padding=8)
        rf.pack(fill="x", pady=4)
        self.train_nrmse_lbl = ttk.Label(rf, text="Training: --", font=("TkDefaultFont", 11, "bold"))
        self.train_nrmse_lbl.pack(anchor="w")
        self.single_nrmse_lbl = ttk.Label(rf, text="Single test: --", font=("TkDefaultFont", 11, "bold"))
        self.single_nrmse_lbl.pack(anchor="w")

        self.status_lbl = ttk.Label(left, text="Idle")
        self.status_lbl.pack(anchor="w", pady=6)

        # --- right side: plots + log ---
        right = ttk.Frame(self, padding=4)
        right.pack(side="right", fill="both", expand=True)

        top = ttk.Notebook(right)
        top.pack(fill="both", expand=True)
        self.train_nb = ttk.Notebook(top)
        self.single_nb = ttk.Notebook(top)
        top.add(self.train_nb, text="Training graphs")
        top.add(self.single_nb, text="Single test graph")
        self.top_nb = top

        self.log = tk.Text(right, height=9, state="disabled", wrap="word")
        self.log.pack(fill="x", pady=(4, 0))

    # ---------------- helpers ----------------
    def _set_busy(self, busy, text=None):
        self.busy = busy
        state = "disabled" if busy else "normal"
        for b in (self.train_btn, self.loo_btn, self.single_btn):
            b.config(state=state)
        self.status_lbl.config(text=text if text else ("Running..." if busy else "Idle"))

    def _log(self, s):
        self.log.config(state="normal")
        self.log.insert("end", s)
        self.log.see("end")
        self.log.config(state="disabled")

    def _clear_nb(self, nb):
        for tab in nb.tabs():
            nb.nametowidget(tab).destroy()

    def _add_figure(self, nb, fig, nrmse):
        frame = ttk.Frame(nb)
        title = fig.axes[0].get_title() if fig.axes else "Figure"
        if title.startswith("Experiment") and nrmse is not None:
            ttk.Label(frame, text=f"{title}   |   NRMSE: {nrmse}%",
                      font=("TkDefaultFont", 10, "bold")).pack(anchor="w", padx=4)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        nb.add(frame, text=title if title else "Figure")
        nb.select(frame)

    # ---------------- run a function in a worker thread ----------------
    def _run(self, func, kind, *args):
        if self.busy:
            return
        self._set_busy(True)
        self.latest_nrmse = None
        if kind == "train":
            self._clear_nb(self.train_nb)
            self.train_nrmse_lbl.config(text="Training: --")
            self.top_nb.select(self.train_nb)
        else:
            self._clear_nb(self.single_nb)
            self.single_nrmse_lbl.config(text="Single test: --")
            self.top_nb.select(self.single_nb)

        def worker():
            old_out, old_show = sys.stdout, plt.show
            sys.stdout = _QueueWriter(self.q)

            def gui_show(*a, **k):          # replaces plt.show() so graphs go to the GUI
                fig = plt.gcf()
                self.q.put(("fig", kind, fig))
                plt.close(fig)
            plt.show = gui_show
            try:
                result = func(*args)
                self.q.put(("done", kind, result))
            except Exception:
                self.q.put(("error", traceback.format_exc()))
            finally:
                sys.stdout, plt.show = old_out, old_show

        threading.Thread(target=worker, daemon=True).start()

    # ---------------- button callbacks ----------------
    def on_train(self):
        try:
            kan_width = ast.literal_eval(self.width_var.get())
            kan_grid = int(self.grid_var.get())
            kan_k = int(self.k_var.get())
            if not isinstance(kan_width, list):
                raise ValueError("kan_width must be a list, e.g. [4,8,8,1]")
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return
        self._run(perform_training, "train", kan_width, kan_grid, kan_k)

    def on_loo(self):
        self._run(perform_LOO_training, "train")

    def on_single(self):
        if "model" not in gui_state:
            messagebox.showwarning("No model", "Train a model first (single_test uses the last trained model).")
            return
        try:
            exp_num = int(self.exp_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Experiment # must be an integer.")
            return
        self._run(single_test, "single",
                  gui_state["model"], exp_num, gui_state["time_mx_mn"], gui_state["PCD_mx_mn"],
                  gui_state["volt_mx_mn"], gui_state["bias_mx_mn"], gui_state["load_mx_mn"])

    # ---------------- main-thread queue polling ----------------
    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "log":
                    self._log(msg[1])
                    m = re.search(r"Test NRMSE:\s*([-\d.]+)%", msg[1])
                    if m:
                        self.latest_nrmse = m.group(1)
                        self.status_lbl.config(text=f"Running... (latest NRMSE {self.latest_nrmse}%)")
                elif msg[0] == "fig":
                    nb = self.train_nb if msg[1] == "train" else self.single_nb
                    self._add_figure(nb, msg[2], self.latest_nrmse)
                elif msg[0] == "done":
                    if msg[1] == "train":
                        if self.latest_nrmse is not None:
                            self.train_nrmse_lbl.config(text=f"Training (last test set): {self.latest_nrmse}%")
                    else:
                        self.single_nrmse_lbl.config(text=f"Single test: {msg[2]}%")
                    self._set_busy(False, "Done")
                elif msg[0] == "error":
                    self._log(msg[1])
                    self._set_busy(False, "Error (see log)")
                    messagebox.showerror("Error", msg[1].strip().splitlines()[-1])
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    KANApp().mainloop()
