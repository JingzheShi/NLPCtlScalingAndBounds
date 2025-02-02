import torch
import numpy as np


FILE_LIST = [
    '8_eigvals.pt',
    '12_eigvals.pt',
    '16_eigvals.pt',
    '24_eigvals.pt',
    '32_eigvals.pt',
    
    '48_eigvals.pt',
    '64_eigvals.pt',
    '128_eigvals.pt',
    '256_eigvals.pt',
    '512_eigvals.pt',
    
    
]

ctl_list = [8,12,16,24,32,48,64,128,256,512]

FILE_TAG = [f'{ctl}ctl' for ctl in ctl_list]


from matplotlib import pyplot as plt
plt.figure()

# dummy. just to keep the code running without error.
X_PRED_MIN = [
    16,16,16,16,16,16,16,16,16,16,
]

X_PRED_MAX = [
    16+1,16+1,16+1,16+1,16+1,16+1,16+1,16+1,16+1,16+1,
]
# dummy. just to keep the code running without error.

y_list = []
colors = []
COLOR_LIST = ['r','orange','y','g','b','purple']

COLOR_LIST = plt.cm.viridis(np.linspace(0,0.97,len(FILE_LIST)))


THRESHOLD = 0.0035




pred_id_list = list()

prev_eigval = None
prev_tag = None
for idx, (file, tag, pred, pred_max) in enumerate(zip(FILE_LIST, FILE_TAG, X_PRED_MIN, X_PRED_MAX)):
    
    eig_vals = torch.load(file) # numpy.ndarray.
    lambda_max = eig_vals[0]
    eig_vals = eig_vals / lambda_max
    # find the last index that is larger than THRESHOLD.
    pred = np.where(eig_vals > THRESHOLD)[0][-1]
    pred_id_list.append(pred)
    
    # plot all eigenvalues on one graph.
    x_axis = np.array(range(1,1+len(eig_vals)))
    y_pred = eig_vals[pred-1]
    y_pred_min = eig_vals[pred_max-1]
    y_list.append((y_pred,y_pred_min))
    print((y_pred,y_pred_min,tag,pred))
    
    
    if idx != 0:
        plt.plot(x_axis, eig_vals/prev_eigval - 1, label=f"{tag}/{prev_tag}", color=COLOR_LIST[len(y_list)-1])
        colors.append(plt.gca().lines[-1].get_color())
        
    prev_eigval = eig_vals
    prev_tag = tag
    
    

ctls = np.array(ctl_list)
pred_id_list = np.array(pred_id_list)
# use ID = C_0 + C/ctl^gamma to fit the data.

def fitted_func(ctl, C_0, C, gamma):
    return C_0 + C / (ctl) ** gamma

from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
# fit curve:

popt, pcov = curve_fit(fitted_func, ctls, pred_id_list, p0=[3000,-30000,1])

C_0, C, gamma= popt
dC_0, dC, dgamma= np.sqrt(np.diag(pcov))
# calculate R².
pred_id_list_pred = fitted_func(ctls, C_0, C, gamma)
r2 = r2_score(pred_id_list, pred_id_list_pred)
print(f'C_0={C_0:.3f}±{dC_0:.3f}, C={C:.3f}±{dC:.3f}, gamma={gamma:.3f}±{dgamma:.3f}, R²={r2:.5f}')


[8,12,16,24,32,48,64,128,256,512]
Ctl_Loss=np.array([8,12,16,24,32,48,64,128,256,384,512,768])
CE_Loss=np.array([3.9531,3.5390,3.28283,2.99618,2.83873,2.65246,2.53125,2.36169,2.21765,2.14698,2.12716,2.08389])

corresponding_CE_Loss = np.array([
    3.9531,3.5390,3.2883,
    2.99618,2.83873,2.65246,2.53125,2.36169,2.12716,2.08389
]) # corresponding CE Loss for each ctl.

THRESHOLD_TO_ID = dict()

N_TO_ID = dict()

def try_threshold(threshold, N=5, n_to_id = N_TO_ID):
    pred_list = list()
    for file, tag in zip(FILE_LIST, FILE_TAG):
        eig_vals = torch.load(file) # numpy.ndarray.
        lambda_max = eig_vals[0]
        eig_vals = eig_vals / lambda_max
        eig_val_log = np.log(eig_vals)
        # eig_val_log[eig_val_log < 1] = 0
        # eig_val_log[:1]=0
        eig_val_log[N:]=0
        pred = np.sum(eig_val_log)#**2)**1# np.sum(np.log(eig_vals))
        print((tag,pred))
        pred_list.append(pred)
    pred_list = np.array(pred_list)
    # fit curve:
    popt, pcov = curve_fit(fitted_func, ctls, pred_list, p0=[3000,-30000,1])
    C_0, C, gamma= popt
    dC_0, dC, dgamma = np.sqrt(np.diag(pcov))
    # calculate R².
    pred_list_pred = fitted_func(ctls, C_0, C, gamma)

    r2 = r2_score(pred_list, pred_list_pred)
    print(f'Thr: {threshold}, C_0={C_0:.3f}±{dC_0:.3f}, C={C:.3f}±{dC:.3f}, gamma={gamma:.5f}±{dgamma:.5f}, R²={r2:.5f}')
    # fit corresponding_CE_Loss = k * pred_list + b.
    # print(len(pred_list),len(corresponding_CE_Loss))
    k, b = np.polyfit(pred_list, corresponding_CE_Loss, 1)
    
    r2 = r2_score(corresponding_CE_Loss, k * np.array(pred_list) + b)
    N_TO_ID[N] = dict(
        pred_list_pred = pred_list,
        k = k,
        b = b,
        r2 = r2
    )
    n_to_id[N] = dict(
        pred_list_pred = pred_list,
        k = k,
        b = b,
        r2 = r2
    )
    THRESHOLD_TO_ID[threshold] = dict(
        pred_list_pred = pred_list,
        k = k,
        b = b,
        r2 = r2
    )
    print(f'Fit CE with ID, k={k:.3f}, b={b:.3f}, -b/k = {-b/k:.3f} R²={r2:.5f}')
    



# plt.xscale('log')
plt.yscale('log')
# plt.ylim(2*10**-5,1.01)
# draw y = 0.025.

THRESHOLDS = [0.002,0.0025,0.003,0.0035,0.004,0.005,0.0075,0.01,0.0125,0.015,0.02,0.025,0.03,0.05,0.075,0.1,0.125,0.15,0.2,0.25]


LINESTYLE = ':'


# plt.xlim(0,55)
# plt.ylim(2.5e-2, 1.1)
plt.legend()
plt.xlabel('Index of Eigenvalue')
plt.ylabel('$\\frac{(Eigenvalue / Max Eigenvalue)_{ctl1}}{(Eigenvalue / Max Eigenvalue)_{ctl2}} - 1$')

plt.title('Relative Increment of Eigenvalues between Different Context Lengths')
plt.savefig('eig_vals_increment.svg')#,dpi=250)#, dpi=300)

# [try_threshold(threshold) for threshold in [0.002,0.0025,0.003,0.0035,0.004,0.005,0.0075,0.01,0.0125,0.015,0.02,0.025,0.03,0.05,0.075,0.1,0.125,0.15,0.2,0.25]]

NS = [3,4,5,7]#,20,50,100,]
def draw_for_NS(NS=NS, savename='CE_vs_ID_N.jpg', title='Cross Entropy Loss vs Measured Intrinsic Dimension for certain Threshold', N_TO_ID = dict()):
    [try_threshold(0.002,n, N_TO_ID) for n in NS]

    # THRESHOLD_TO_ID: key is threshold, value is a dict:
    # {
    #     pred_list_pred: pred_list_pred,
    #     k: k,
    #     b: b,
    #     r2: r2
    # }
    # now, draw CE_Loss vs ID, and draw the fitted line, for each threshold as a different color.

    plt.figure()
    keys_list = list(N_TO_ID.keys())
    colors = plt.cm.viridis(np.linspace(0,0.97,len(keys_list)))
    for idx,N in enumerate(keys_list):
        color = colors[idx]
        plt.scatter(N_TO_ID[N]['pred_list_pred'], corresponding_CE_Loss, color=color)
        pred_list_pred = N_TO_ID[N]['pred_list_pred']
        k = N_TO_ID[N]['k']
        b = N_TO_ID[N]['b']
        r2 = N_TO_ID[N]['r2']
        
        line_x = np.array(pred_list_pred)
        line_x = np.sort(line_x)
        line_x_mean = np.mean(line_x)
        line_x[-1] = (2.0-b)/k
        line_y = k * line_x + b
        plt.plot(line_x, line_y, label=f'N={N}, R²={r2:.3f}', color=color)
    plt.xlabel('sum(log(relative Eigen Value)) in first N dimensions')
    # plt.xscale('log')
    plt.ylabel('Cross Entropy Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(savename)#,dpi=250)#, dpi=300)
    plt.close()
    return N_TO_ID

N_TO_ID = dict()
draw_for_NS(NS=[3,4,5,6], savename='CE_vs_ID_N_1.svg', title='Cross Entropy Loss vs Measured Intrinsic Dimension for certain Threshold\n (part 1)', N_TO_ID = dict())
draw_for_NS(NS=[10,15,25,35,45,], savename='CE_vs_ID_N_2.svg', title='Cross Entropy Loss vs Measured Intrinsic Dimension for certain Threshold\n (part 2)', N_TO_ID = dict())
draw_for_NS(NS=[50,80,100,120,150], savename='CE_vs_ID_N_3.svg', title='Cross Entropy Loss vs Measured Intrinsic Dimension for certain Threshold\n (part 3)', N_TO_ID = dict())
draw_for_NS(NS=[200,300,450,600,750], savename='CE_vs_ID_N_4.svg', title='Cross Entropy Loss vs Measured Intrinsic Dimension for certain Threshold\n (part 4)', N_TO_ID = dict())
draw_for_NS(NS=[1000,1300,1500,1800,2000], savename='CE_vs_ID_N_5.svg', title='Cross Entropy Loss vs Measured Intrinsic Dimension for certain Threshold\n (part 5)', N_TO_ID = dict())

N_TO_ID['-CE_loss'] = {
    'pred_list_pred': -corresponding_CE_Loss,
}

# C_小标0用小写0写出来是：C₀

print(N_TO_ID.keys())

keys = list(N_TO_ID.keys())
last_key = keys[-1]
keys = [last_key] + keys[:-1]
keys=keys[:1]+keys[2:]


corr_dict = dict()
for i in range(len(keys)):
    for j in range(i+1):
        N1 = keys[i]
        N2 = keys[j]
        pred_list_pred1 = np.array(N_TO_ID[N1]['pred_list_pred'])
        pred_list_pred2 = np.array(N_TO_ID[N2]['pred_list_pred'])
        # calculate correlation.
        corr = np.corrcoef(pred_list_pred1, pred_list_pred2)[0,1]
        print(f'N1={N1}, N2={N2}, corr={corr:.5f}')
        corr_dict[(N1,N2)] = float(corr)
        corr_dict[(N2,N1)] = float(corr)

# draw a heatmap.

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# do not create a dataframe.

# do not do: sns.heatmap(df.astype(float), annot=True, cmap='coolwarm', fmt=".3f")

# this is wrong:
# df = pd.DataFrame(corr_dict, index=keys, columns=keys)
df = pd.DataFrame(0, index=keys, columns=keys)

# Fill the DataFrame with correlation values
for (i, j), corr in corr_dict.items():
    df.loc[i, j] = corr
print(df)
plt.figure(figsize=(12, 10))
ax = plt.gca()
ax.set_yscale('linear')
ax.set_xscale('linear')
min_val = df.min().min()  # Get minimum value in the DataFrame
if min_val < 0.975:
    print(f"Found value(s) below threshold: minimum is {min_val}")
    assert False, f"Values below 0.97 detected! Minimum value: {min_val}"
sns.heatmap(df, 
            annot=False,
            cmap='YlGn',
            vmin=0.98,
            vmax=1.0,
            linewidths=0.5,     # Add thin lines
            linecolor='gray'    # Make lines gray
)
# correct should be:

plt.xticks(rotation=45, ha='right', fontsize=15)  # Increase x-axis label size
plt.yticks(fontsize=15)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.title('Correlation between - CE Loss and sum(log(rel_eig_val)) for different N', fontsize=18)
plt.savefig('Correlation_between_different_N.svg')