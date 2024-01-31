# --- Funtions --- #

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def read_files(method, dimension, h_band, l_band):


    original_high = f'datasamples_{dimension}/input_hr_{dimension}.csv'
    original_low = f'datasamples_{dimension}/input_lr_{dimension}.csv'
    recontraction_high = f'datasamples_{dimension}/high_recon_{method}-{dimension}.csv'
    recontraction_low = f'datasamples_{dimension}/low_recon_{method}-{dimension}.csv'

    data_h = pd.read_csv(original_high, header=None).values.astype(float)
    data_recon_h = pd.read_csv(recontraction_high, header=None).values.astype(float)
    data_l = pd.read_csv(original_low, header=None).values.astype(float)
    data_recon_l = pd.read_csv(recontraction_low, header=None).values.astype(float)

    if dimension == 145: dimension = 150

    data_recon_h = data_recon_h.reshape( h_band,dimension,dimension)
    data_recon_h = np.swapaxes(data_recon_h, 0, 1)
    data_h = data_h.reshape(h_band,dimension,dimension)
    data_h = np.swapaxes(data_h, 0, 1)
    data_recon_l = data_recon_l.reshape(l_band,dimension,dimension)
    data_recon_l = np.swapaxes(data_recon_l, 0, 1)
    data_l = data_l.reshape(l_band,dimension,dimension)
    data_l = np.swapaxes(data_l, 0, 1)
    

    return data_h, data_recon_h, data_l, data_recon_l


def random_init(h_band):

    # Ensure there are at least 6 bands available
    if h_band >= 6:
        # Choose two random values with a distance over 5
        random_band1, random_band2 = random.sample(range(h_band), 2)
        while abs(random_band1 - random_band2) <= 5:
            random_band1, random_band2 = sorted(random.sample(range(h_band), 2))

        return random_band1, random_band2
    else:
        raise ValueError("Not enough bands available.")


def resid(data,data_recon):

    error = abs(data- data_recon)
    error_max = np.max(data)

    # Normalize the high_error array
    norm_error = error/ error_max 

    return norm_error

def calculate_psnr(original, reconstructed):

    # Calculate MSE
    mse = np.mean((original - reconstructed) ** 2)

    # Calculate the peak value Lm (assuming data is in the range [0, 1])
    Lm = np.max(original)

    # Calculate PSNR
    psnr = 10 * np.log10(Lm**2 / mse)

    return psnr

def calculate_sam(testing_spectrum, reference_spectrum):
    # Calculate the dot product of testing and reference spectra
    dot_product = np.sum(testing_spectrum * reference_spectrum)

    # Calculate the magnitude of testing and reference spectra
    magnitude_testing = np.linalg.norm(testing_spectrum)
    magnitude_reference = np.linalg.norm(reference_spectrum)

    # Calculate the cosine of the angle between the spectra
    cosine_theta = dot_product / (magnitude_testing * magnitude_reference)

    # Calculate the angle in radians and convert it to degrees
    angle_rad = np.arccos(np.clip(cosine_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def calculate_sam_for_all_bands(data_testing, data_reference):
    # Get the number of spectral bands
    num_bands = data_testing.shape[0]

    # Initialize an array to store SAM values for each band
    sam_values = np.zeros(num_bands)

    # Calculate SAM for each spectral band
    for i in range(num_bands):
        spectrum_testing = data_testing[i].flatten()
        spectrum_reference = data_reference[i].flatten()
        sam_values[i] = calculate_sam(spectrum_testing, spectrum_reference)

    return sam_values

def show_results(data_recon_h, data_h,data_recon_l, data_l):

    psnr_value_h = calculate_psnr(data_recon_h, data_h)
    psnr_value_l = calculate_psnr(data_recon_l, data_l)

    # Prints
    print(f"PSNR-High: {psnr_value_h:.2f} dB")
    print(f"PSNR-Low: {psnr_value_l:.2f} dB")

    print('===================')
    ############################################
   
    sam_values_high = calculate_sam_for_all_bands(data_recon_h, data_h)
    sam_values_low = calculate_sam_for_all_bands(data_recon_l, data_l)

    # Find the Average
    average_sam_high = np.mean(sam_values_high)
    average_sam_low = np.mean(sam_values_low)

    # Prints
    print(f"Average SAM-High: {average_sam_high:.2f} degrees")
    print(f"Average SAM-Low: {average_sam_low:.2f} degrees")

    return psnr_value_h,psnr_value_l, average_sam_high, average_sam_low


def show_plots(data, data_recon, band1, band2, Name):

    norm = resid(data,data_recon)

    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.25)

    # Create a 2x4 subplot
    plt.subplot(2, 3, 1)
    plt.imshow(data[:,band1,:])
    plt.title(f'{Name} Res. Original - Band {band1}')

    plt.subplot(2, 3, 2)
    plt.imshow(data_recon[:,band1,:])
    plt.title(f'{Name} Res. Reconstracted - Band {band1}')

    #Residual
    plt.subplot(2, 3, 3)
    plt.imshow(norm[:,band1,:],cmap='cividis')
    plt.title(f'Residual - Band {band1}')

    plt.subplot(2, 3, 4)
    plt.imshow(data[:,band2,:])
    plt.title(f'{Name} Res. Original - Band {band2}')

    plt.subplot(2, 3, 5)
    plt.imshow(data_recon[:,band2,:])
    plt.title(f'{Name} Res. Reconstracted - Band {band2}')

    #Residual
    plt.subplot(2, 3, 6)
    plt.imshow(norm[:,band2,:],cmap='cividis')
    plt.title(f'Residual - Band {band2}')

    return

