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
    
    
    if idx >= 0:
        plt.plot(x_axis, eig_vals, label=tag, color=COLOR_LIST[len(y_list)-1])
        colors.append(plt.gca().lines[-1].get_color())
        
    prev_eigval = eig_vals
    
    

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
])

THRESHOLD_TO_ID = dict()


def try_threshold(threshold):
    pred_list = list()
    for file, tag in zip(FILE_LIST, FILE_TAG):
        eig_vals = torch.load(file) # numpy.ndarray.
        lambda_max = eig_vals[0]
        eig_vals = eig_vals / lambda_max
        # find the last index that is larger than THRESHOLD.
        pred = np.where(eig_vals > threshold)[0][-1]
        eig_val_log = np.log(eig_vals)
        # eig_val_log[eig_val_log < 1] = 0
        eig_val_log[:1]=0
        eig_val_log[5:]=0
        # pred = np.sum(eig_val_log)#**2)**1# np.sum(np.log(eig_vals))
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

plt.legend()
# x轴标签: Index of Eigenvalue
# y轴标签: Eigenvalue / Max Eigenvalue
plt.xlabel('Index of Eigenvalue')
plt.ylabel('Eigenvalue / Max Eigenvalue')
plt.title('Eigenvalues of Mid-Features')
plt.savefig('eig_vals_log.svg')#,dpi=250)#, dpi=300)

[try_threshold(threshold) for threshold in [0.002,0.0025,0.003,0.0035,0.004,0.005,0.0075,0.01,0.0125,0.015,0.02,0.025,0.03,0.05,0.075,0.1,0.125,0.15,0.2,0.25]]


# THRESHOLD_TO_ID: key is threshold, value is a dict:
# {
#     pred_list_pred: pred_list_pred,
#     k: k,
#     b: b,
#     r2: r2
# }
# now, draw CE_Loss vs ID, and draw the fitted line, for each threshold as a different color.

plt.figure()

keys_list = list(THRESHOLD_TO_ID.keys())

colors = plt.cm.viridis(np.linspace(0,0.97,len(keys_list[:1+len(keys_list)//4])))
for idx,threshold in enumerate(keys_list[:1+len(keys_list)//4]):
    color = colors[idx]

    plt.scatter(THRESHOLD_TO_ID[threshold]['pred_list_pred'], corresponding_CE_Loss, color=color)
    pred_list_pred = THRESHOLD_TO_ID[threshold]['pred_list_pred']
    k = THRESHOLD_TO_ID[threshold]['k']
    b = THRESHOLD_TO_ID[threshold]['b']
    r2 = THRESHOLD_TO_ID[threshold]['r2']
    
    # y = kx + b. ymin = 2.0, y max = 4.0.
    x_1 = (2.0 - b) / k
    x_2 = (4.0 - b) / k
    
    plt.plot([x_1, x_2], [2.0, 4.0], label=f'Thres:{threshold}, R²={r2:.4f}', color=color, linestyle='--')

    
    
    # plt.plot(pred_list_pred, k * np.array(pred_list_pred) + b, label=f'Threshold={threshold}, R²={r2:.5f}')
plt.xlabel('log(States)')
# plt.xscale('log')
plt.ylabel('Cross Entropy Loss')
plt.title('Cross Entropy Loss vs Measured Intrinsic Dimension for certain Threshold\n (part 4)')
plt.legend()
plt.savefig('CE_vs_ID_1.jpg')#,dpi=250)#, dpi=300)

plt.figure()



plt.figure()

keys_list = list(THRESHOLD_TO_ID.keys())

colors = plt.cm.viridis(np.linspace(0,0.97,len(keys_list[1+len(keys_list)//4:len(keys_list)//2])))
for idx,threshold in enumerate(keys_list[1+len(keys_list)//4:len(keys_list)//2]):
    color = colors[idx]

    plt.scatter(THRESHOLD_TO_ID[threshold]['pred_list_pred'], corresponding_CE_Loss, color=color)
    pred_list_pred = THRESHOLD_TO_ID[threshold]['pred_list_pred']
    k = THRESHOLD_TO_ID[threshold]['k']
    b = THRESHOLD_TO_ID[threshold]['b']
    r2 = THRESHOLD_TO_ID[threshold]['r2']
    
    # y = kx + b. ymin = 2.0, y max = 4.0.
    x_1 = (2.0 - b) / k
    x_2 = (4.0 - b) / k
    plt.plot([x_1, x_2], [2.0, 4.0], label=f'Thres:{threshold}, R²={r2:.4f}', color=color, linestyle='--')

    
    
    # plt.plot(pred_list_pred, k * np.array(pred_list_pred) + b, label=f'Threshold={threshold}, R²={r2:.5f}')
plt.xlabel('log(States)')
# plt.xscale('log')
plt.ylabel('Cross Entropy Loss')
plt.title('Cross Entropy Loss vs Measured Intrinsic Dimension for certain Threshold\n(part 3)')
plt.legend()
plt.savefig('CE_vs_ID_2.jpg')#,dpi=250)#, dpi=300)

plt.figure()

keys_list = list(THRESHOLD_TO_ID.keys())

colors = plt.cm.viridis(np.linspace(0,0.97,len(keys_list[1+len(keys_list)//2:1+3*len(keys_list)//4])))
for idx,threshold in enumerate(keys_list[1+len(keys_list)//2:1+3*len(keys_list)//4]):
    color = colors[idx]

    plt.scatter(THRESHOLD_TO_ID[threshold]['pred_list_pred'], corresponding_CE_Loss, color=color)
    pred_list_pred = THRESHOLD_TO_ID[threshold]['pred_list_pred']
    k = THRESHOLD_TO_ID[threshold]['k']
    b = THRESHOLD_TO_ID[threshold]['b']
    r2 = THRESHOLD_TO_ID[threshold]['r2']
    
    # y = kx + b. ymin = 2.0, y max = 4.0.
    x_1 = (2.0 - b) / k
    x_2 = (4.0 - b) / k
    plt.plot([x_1, x_2], [2.0, 4.0], label=f'Thres:{threshold}, R²={r2:.4f}', color=color, linestyle='--')
    
    
    # plt.plot(pred_list_pred, k * np.array(pred_list_pred) + b, label=f'Threshold={threshold}, R²={r2:.5f}')
plt.xlabel('log(States)')
# plt.xscale('log')
plt.ylabel('Cross Entropy Loss')
plt.title('Cross Entropy Loss vs Measured Intrinsic Dimension for certain Threshold\n(part 2)')
plt.legend()
plt.savefig('CE_vs_ID_3.jpg')#,dpi=250)#, dpi=300)


plt.figure()

keys_list = list(THRESHOLD_TO_ID.keys())

colors = plt.cm.viridis(np.linspace(0,0.97,len(keys_list[1+3*len(keys_list)//4:4*len(keys_list)//4])))
for idx,threshold in enumerate(keys_list[1+3*len(keys_list)//4:4*len(keys_list)//4]):
    color = colors[idx]

    plt.scatter(THRESHOLD_TO_ID[threshold]['pred_list_pred'], corresponding_CE_Loss, color=color)
    pred_list_pred = THRESHOLD_TO_ID[threshold]['pred_list_pred']
    k = THRESHOLD_TO_ID[threshold]['k']
    b = THRESHOLD_TO_ID[threshold]['b']
    r2 = THRESHOLD_TO_ID[threshold]['r2']
    
    # y = kx + b. ymin = 2.0, y max = 4.0.
    x_1 = (2.0 - b) / k
    x_2 = (4.0 - b) / k
    plt.plot([x_1, x_2], [2.0, 4.0], label=f'Thres:{threshold}, R²={r2:.4f}', color=color, linestyle='--')
    
    
    # plt.plot(pred_list_pred, k * np.array(pred_list_pred) + b, label=f'Threshold={threshold}, R²={r2:.5f}')
plt.xlabel('log(States)')
# plt.xscale('log')
plt.ylabel('Cross Entropy Loss')
plt.title('Cross Entropy Loss vs Measured Intrinsic Dimension for certain Threshold\n(part 1)')
plt.legend()
plt.savefig('CE_vs_ID_4.jpg')#,dpi=250)#, dpi=300)