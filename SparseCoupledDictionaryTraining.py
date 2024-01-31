from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from numpy import genfromtxt
from math import sqrt
import pandas as pd
from numpy.linalg import inv
from args import get_opts
from CDLOps import CDL, updateCDL
import time
import matplotlib.pyplot as plt
import os

def BFGS_update(H, s, y):
    """
    Perform a BFGS (Broyden–Fletcher–Goldfarb–Shanno) update on the approximation matrix H.

    Parameters:
    H (numpy.ndarray): The current approximation to the inverse Hessian matrix.
    s (numpy.ndarray): The difference in the optimization variable (x_k+1 - x_k).
    y (numpy.ndarray): The difference in the gradients (grad_f(x_k+1) - grad_f(x_k)).

    Returns:
    numpy.ndarray: The updated approximation to the inverse Hessian matrix.
    """
    
    s = s.reshape(-1, 1)  # Ensure s is a column vector
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    sy_dot = y.T @ s
    if np.abs(sy_dot) < 1e-10:  # Check for division by a very small number
        return np.zeros_like(H)  # Skip update if sy_dot is too small original was H

    rho = 1.0 / sy_dot
    Hy = H @ y
    H_new = H - rho * (Hy @ Hy.T) + rho * (s @ s.T)

    if np.any(np.isnan(H_new)) or np.any(np.isinf(H_new)):
        return np.zeros_like(H)  # Skip update if resulting H is invalid original was H
    return H_new


def compute_gradient_Dh(Dh, Wh, Sh):
    """
    Compute the gradient of the objective function with respect to the dictionary matrix Dh.

    Parameters:
    Dh (numpy.ndarray): The dictionary matrix Dh, which is part of the model parameters.
    Wh (numpy.ndarray): The sparse coding matrix Wh, representing the coefficients in the sparse representation.
    Sh (numpy.ndarray): The high-resolution data matrix, representing the data to be approximated.

    Returns:
    numpy.ndarray: The gradient of the objective function with respect to Dh.
    """
    
    return -(Sh - np.dot(Dh, Wh)) @ Wh.T


def compute_gradient_Dl(Dl, Wl, Sl):
    """
    Compute the gradient of the objective function with respect to the dictionary matrix Dl.

    Parameters:
    Dl (numpy.ndarray): The dictionary matrix Dl, which is a part of the model parameters.
    Wl (numpy.ndarray): The sparse coding matrix Wl, representing the coefficients in the sparse representation.
    Sl (numpy.ndarray): The low-resolution data matrix, representing the data that is being approximated.

    Returns:
    numpy.ndarray: The gradient of the objective function with respect to Dl.
    """
    return -(Sl - np.dot(Dl, Wl)) @ Wl.T

def normD(dictin):
    """
    Normalize the columns of the dictionary matrix to have unit norm, scaling them to a range between [0, 1].

    Parameters:
    dictin (numpy.ndarray): The input dictionary matrix, where each column represents a dictionary atom.

    Returns:
    numpy.ndarray: The normalized dictionary matrix with each column having a unit norm.
    """
    # Calculate normalization factor for each column
    norm_factor = 1 / np.sqrt(np.sum(np.multiply(dictin, dictin), axis=0))

    # Normalize each column of the dictionary and return the result
    return np.dot(dictin, np.diag(norm_factor))


def show_loss(loss_h, loss_l, l_it, path):
    """
    Visualizes the loss for high and low-resolution data.
    :param loss_h: List of high-resolution losses.
    :param loss_l: List of low-resolution losses.
    :param l_it: Iteration number to mark on the plot.
    :param path: Path where to save the plot.
    """
    plt.figure(figsize=(12, 6))

    # Plot for Loss High
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.plot(loss_h)
    plt.axvline(x=l_it, color='r', linestyle='--')  # Add vertical line
    plt.title('Loss High')
    # Add text for loss_h at iteration l_it
    plt.text(l_it, loss_h[l_it], f"{loss_h[l_it]:.2f}", color='blue', fontsize=10)

    # Plot for Loss Low
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.plot(loss_l)
    plt.axvline(x=l_it, color='r', linestyle='--')  # Add vertical line
    plt.title('Loss Low')
    # Add text for loss_l at iteration l_it
    plt.text(l_it, loss_l[l_it], f"{loss_l[l_it]:.2f}", color='blue', fontsize=10)

    plt.savefig(os.path.join(path, "loss_plot_.png"))
    plt.close()


def run_script():
    """
    Runs the main script for optimization and visualization.
    :param opts: Options and parameters for the script.
    """
    
    # Initilization
    #------------------------------------------------------------------------------------
    imageN = opts.imageN
    dictsize = opts.dictsize
    bands_h_N = opts.bands_h
    bands_l_N = opts.bands_l
    c1 = opts.c1
    c2 = opts.c2
    c3 = opts.c3
    maxbeta = opts.maxbeta
    delta = opts.delta
    beta = opts.beta
    beta_ = opts.sec_lr
    lamda = opts.lamda
    train_iter = opts.n_iter
    wind = opts.window

    data_h = genfromtxt(opts.inputhigh, delimiter=',')
    data_l = genfromtxt(opts.inputlow, delimiter=',')

    dict_h = data_h[:, 0:dictsize]
    dict_l = data_l[:, 0:dictsize]

    cdl = CDL(data_h, data_l, dictsize, imageN)

    phi_h = np.zeros(dictsize)
    phi_l = np.zeros(dictsize)

    Hh = np.eye(dict_h.size)  # Initialize inverse Hessian approximation for dict_h
    Hl = np.eye(dict_l.size)  # Initialize inverse Hessian approximation for dict_l

    loss_l = []
    loss_h = []
    best_l = np.inf
    best_h = np.inf
    best = np.inf
#-----------------------------------------END----------------------------------------




# Main Loop
#------------------------------------------------------------------------------------
    for k in range(train_iter):  # Begin training loop for a specified number of iterations
        start_time = time.time()  # Record the start time of the iteration

        # Transpose dictionaries for high and low resolutions
        dict_ht = np.transpose(dict_h)  
        dict_lt = np.transpose(dict_l)

        # Compute matrices needed for updating the dictionary
        dtdh = np.dot(np.transpose(dict_h), dict_h) + (c1 + c3) * np.eye(np.transpose(dict_h).shape[0])
        dtdhinv = inv(dtdh)  # Invert the computed matrix for high resolution dictionary

        dtdl = np.dot(np.transpose(dict_l), dict_l) + (c2 + c3) * np.eye(np.transpose(dict_l).shape[0])
        dtdlinv = inv(dtdl)  # Invert the computed matrix for low resolution dictionary

        # Update Coupled Dictionary Learning (CDL) with the new matrices
        cdl = updateCDL(cdl, dict_ht, dict_lt, dtdhinv, dtdlinv, c1, c2, c3, maxbeta, beta, lamda)

        # Compute gradients for the high and low resolution dictionaries
        grad_Dh = compute_gradient_Dh(dict_h, cdl.wh, data_h).flatten()
        grad_Dl = compute_gradient_Dl(dict_l, cdl.wl, data_l).flatten()

        # Store the current state of the dictionaries for later use in BFGS calculations
        dict_h_prev = dict_h.copy()
        dict_l_prev = dict_l.copy()

        # Update dictionary columns for high resolution
        for j in range(dict_h.shape[1]):
            dict_h[:, j] += beta_ * (Hh @ grad_Dh).reshape(dict_h.shape)[:, j]

        # Update dictionary columns for low resolution
        for j in range(dict_l.shape[1]):
            dict_l[:, j] += beta_ * (Hl @ grad_Dl).reshape(dict_l.shape)[:, j]

        # Normalize the updated dictionaries
        dict_h = normD(dict_h)
        dict_l = normD(dict_l)

        # Compute new gradients after dictionary updates
        new_grad_Dh = compute_gradient_Dh(dict_h, cdl.wh, data_h).flatten()
        new_grad_Dl = compute_gradient_Dl(dict_l, cdl.wl, data_l).flatten()

        # Calculate the s and y vectors for the BFGS update
        s_h = dict_h.flatten() - dict_h_prev.flatten()
        y_h = new_grad_Dh - grad_Dh
        Hh = BFGS_update(Hh, s_h, y_h)  # Update high resolution dictionary using BFGS

        s_l = dict_l.flatten() - dict_l_prev.flatten()
        y_l = new_grad_Dl - grad_Dl
        Hl = BFGS_update(Hl, s_l, y_l)  # Update low resolution dictionary using BFGS

        # Calculate error and update loss every 'wind' iterations
        if ~((k + 1) % wind):
            # Calculate reconstruction errors for high and low resolution
            err_h = sqrt(np.sum(np.square(cdl.datain_h - np.dot(dict_h, cdl.wh))) / (bands_h_N * imageN))
            err_l = sqrt(np.sum(np.square(cdl.datain_l - np.dot(dict_l, cdl.wl))) / (bands_l_N * imageN))

            # Append current errors to loss history
            loss_h.append(err_h)
            loss_l.append(err_l)

            # Update best error and save the best reconstruction
            if ((err_h + err_l) / 2) < best:
                best = ((err_h + err_l) / 2)
                best_h = err_h
                best_l = err_l
                high_recon = np.dot(dict_h, cdl.wh)
                low_recon = np.dot(dict_l, cdl.wl)
                l_it = k
                h_it = k

            # Print iteration details
            print(f'Iteration {k}: Error High - {err_h}, Error Low - {err_l}')
        
        # Print time taken for the current iteration
        print(f'Time elapsed for iteration {k}: {time.time() - start_time:.2f} seconds')

    # Display final results
    print('Training Complete\n')
    print(f'Best High Resolution Error: {best_h} at Iteration {l_it}')
    print(f'Best Low Resolution Error: {best_l} at Iteration {h_it}')

    # Save the best reconstructions to CSV files
    pd.DataFrame(high_recon).to_csv(os.path.join(opts.path_out, 'high_recon.csv'), index=False, header=False)
    pd.DataFrame(low_recon).to_csv(os.path.join(opts.path_out, 'low_recon.csv'), index=False, header=False)

    # Function to display loss plots
    show_loss(loss_h, loss_l, l_it, opts.path_out)
#-----------------------------------------END----------------------------------------


def main(args=None):
    global opts

    sec_lr = [0.0001]
    paths_lr = ['sub_matrices/input_lr1.csv','sub_matrices/input_lr2.csv','sub_matrices/input_lr3.csv','sub_matrices/input_lr4.csv','sub_matrices/input_lr5.csv','sub_matrices/input_lr6.csv','sub_matrices/input_lr7.csv','sub_matrices/input_lr8.csv','sub_matrices/input_lr9.csv']
    paths_hr = ['sub_matrices/input_hr1.csv','sub_matrices/input_hr2.csv','sub_matrices/input_hr3.csv','sub_matrices/input_hr4.csv','sub_matrices/input_hr5.csv','sub_matrices/input_hr6.csv','sub_matrices/input_hr7.csv','sub_matrices/input_hr8.csv','sub_matrices/input_hr9.csv']
    count = 0
    for i in range(paths_hr.__len__()):
        count = count+1
        for j in sec_lr:
            sec_lr_dir = f"results/{count}/{j}"
            if not os.path.exists(sec_lr_dir):
                os.makedirs(sec_lr_dir)
                
            opts = get_opts(args)
            opts.inputhigh = paths_hr[i]
            opts.inputlow = paths_lr[i]
            opts.sec_lr = j
            opts.path_out = sec_lr_dir
            run_script()

if __name__ == "__main__":
    main()

