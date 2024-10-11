import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Add the directory containing 'dierssen_qwip_functions' to the system path
sys.path.append(os.path.dirname(__file__))
import dierssen_qwip_functions as qf

# This code calculates QWIP using the functions defined in 'dierssen_qwip_functions'
# Applicable to both hyper- and multispectral data

# DataFrame (df) columns represent wavelengths (e.g. '400'), rows the Rrs values. No other columns and formats are currently handled.
# The QWIP-specific columns are added as new columns.
# Load the df before running the code, e.g. df = pd.read_csv('your_df.csv')

# Currently available multispectral sensor coefficients

sensor_coef = {
    "MODISA": [5.3223151354E-09,-1.3619239245E-05,1.3886726307E-02,-7.0534822746E+00,1.7860303357E+03,-1.8010144488E+05],
    "MODIST": [5.2820302144E-09,-1.3533546593E-05,1.3817487854E-02,-7.0277257404E+00,1.7819361128E+03,-1.7993575351E+05],
    "OLCI-S3A": [5.3756534257E-10,-1.3823299855E-06,1.4217759639E-03,-7.3259519448E-01,1.9025240407E+02,-1.9586875835E+04],
    "OLCI-S3B": [5.2682203391E-10,-1.3545280248E-06,1.3931102207E-03,-7.1787918688E-01,1.8649217615E+02,-1.9204437663E+04],
    "MERIS": [-1.8566475859E-10,5.9630399474E-07,-7.3760076972E-04,4.4214045991E-01,-1.2805555087E+02,1.4733654740E+04],
    "SeaWiFS": [1.3889225090E-08,-3.4666482329E-05,3.4478422505E-02,-1.7081781344E+01,4.2173196004E+03,-4.1487647575E+05],
    "HawkEye": [1.2484460486E-08,-3.1200492734E-05,3.1064705131E-02,-1.5404025698E+01,3.8058443913E+03,-3.7458935612E+05],
    "OCTS": [4.9443860392E-09,-1.2738386299E-05,1.3043106275E-02,-6.6374047608E+00,1.6805141750E+03,-1.6913709216E+05],
    "GOCI": [2.3513883719E-10,-6.3647534573E-07,6.9347646008E-04,-3.8202645304E-01,1.0759457046E+02,-1.2026273574E+04],
    "SGLI": [1.6912426611E-09,-4.9242778829E-06,5.5741262475E-03,-3.0863773616E+00,8.4069664265E+02,-9.0088849592E+04],
    "VIIRS": [1.6399142992E-09,-4.1496452449E-06,4.1742100856E-03,-2.0901181683E+00,5.2296624890E+02,-5.2094618269E+04],
    "NOAA20": [3.8180816885E-10,-1.1345956491E-06,1.2998933044E-03,-7.2752516588E-01,2.0172125993E+02,-2.1958504266E+04],
    "CZCS": [2.5904657929E-08,-6.7326636724E-05,6.9802589933E-02,-3.6085794994E+01,9.3033343084E+03,-9.5665774895E+05],
    "MSI-S2A": [-7.4719642630E-10,1.8794583634E-06,-1.8924227970E-03,9.5069314404E-01,-2.3623941607E+02,2.3384674478E+04],
    "MSI-S2B": [-1.3572501827E-09,3.4546589091E-06,-3.5159381452E-03,1.7855878170E+00,-4.5046398688E+02,4.5327899265E+04],
    "OLI": [-7.5487886903E-09,1.9136260794E-05,-1.9333567648E-02,9.7261770284E+00,-2.4338649740E+03,2.4247497295E+05]
}

# Multispectral sensor wavelengths, define here if needed
sensor_wavelengths = {
    "OLCI-S3A":[412, 443, 490, 510, 560, 620, 665, 673, 681],
    "OLCI-S3B":[412, 443, 490, 510, 560, 620, 665, 673, 681],
    "MSI-S2A":[443, 493, 560, 665],
    "MSI-S2B":[442, 492, 559, 665],
    # your sensors wavelengths
}

def run_qwip_calculations(df, spectral_resolution, wavelengths=None, sensor_coeffs=None):
    """
    Run the QWIP calculation for a given dataframe.

    :param df: input dataframe
    :param spectral_resolution: the spectral_resolution either 'hyper' or 'multi'
    :param wavelengths (optional): multispectral wavelengths - needs to be defined if spectral_resolution == 'multi'
    :param sensor_coeffs (optional): multispectral sensor coeffs - needs to be defined if spectral_resolution == 'multi'
    :return: a tuple containing the updated dataframe, fit1, avw, ndi, ind_400a, ind_500a, ind_600a, index_492, index_665
    """

    # Extract visible Rrs from input DF and calculate the apparent visible wavelength (AVW)
    rrs_vis = qf.extract_columns(df, spectral_resolution, wavelengths)
    avw = qf.calculate_avw(rrs_vis, spectral_resolution, wavelengths, sensor_coeffs)
    
    # calculate NDI / QCI
    ndi,index_492,index_665 = qf.calculate_ndi(rrs_vis.to_numpy(), wavelengths)
    
    # calculate QWIP and scoring
    qwip_score, fit1 = qf.calculate_qwip_score(ndi, avw)
    qwip_flag = qf.flag_qwip_score(qwip_score)
    
    # calc classification indices
    ind_600a, ind_500a, ind_400a = qf.calculate_classification_indices(df, wavelengths)

    # add QWIP values to df
    df['ndi'] = ndi
    df['avw'] = avw
    df['index_492'] = index_492
    df['index_665'] = index_665
    df['qwip_score'] = qwip_score
    df['qwip_flag'] = qwip_flag
    df['ind_400a'] = ind_400a
    df['ind_500a'] = ind_500a
    df['ind_600a'] = ind_600a
   
    return df, fit1, avw, ndi, ind_400a, ind_500a, ind_600a, index_492, index_665

def getQWIPFull(rrsVector, wavenumbers):
    """
    Run the QWIP calculation for a given dataframe.

    :param df: input dataframe
    :param spectral_resolution: the spectral_resolution either 'hyper' or 'multi'
    :param wavelengths (optional): multispectral wavelengths - needs to be defined if spectral_resolution == 'multi'
    :param sensor_coeffs (optional): multispectral sensor coeffs - needs to be defined if spectral_resolution == 'multi'
    :return: a tuple containing the updated dataframe, fit1, avw, ndi, ind_400a, ind_500a, ind_600a, index_492, index_665

    """
    df =[]
    df.append(rrsVector)
    df.append(rrsVector)
    df.append(rrsVector)
    df.append(rrsVector)
    df.append(rrsVector)
    
    df = pd.DataFrame(df, columns=wavenumbers)
    return run_qwip_calculations(df, "hyper", wavenumbers)

def getQWIP(rrsVector, wavenumbers):
    """
    Calculate the QWIP score for a given set of rrsVector and wavenumbers.

    Parameters:
    - rrsVector (list): A list of Rrs values.
    - wavenumbers (list): A list of wavenumber values.

    Returns:
    - qwip_score (Series): A pandas Series containing the QWIP scores.

    """
    qwip_tuple = getQWIPFull(rrsVector, wavenumbers)
    df_out = qwip_tuple[0]
    return df_out["qwip_score"].mean()


# Hyperspectral (assuming 1nm spectral resolution)
def plotQWIP(df, spectral_resolution, wavelengths=None, sensor_coeffs=None):
    """
    Run the QWIP calculation for a given dataframe.

    :param df: input dataframe
    :param spectral_resolution: the spectral_resolution either 'hyper' or 'multi'
    :param wavelengths (optional): multispectral wavelengths - needs to be defined if spectral_resolution == 'multi'
    :param sensor_coeffs (optional): multispectral sensor coeffs - needs to be defined if spectral_resolution == 'multi'
    :return: figure and axis objects
    """
    if sensor_coeffs is not None:
        raise ValueError('This parameter configuration is not supported.')
    
    df_out , fit1, avw, ndi, ind_400a, ind_500a, ind_600a, index_492, index_665 = run_qwip_calculations(df, 
                                                                                                   spectral_resolution, 
                                                                                                   wavelengths=wavelengths)

    # Keep even for multispectral resolution:
    avw_poly = np.arange(400, 631)

    # Uncomment depending on spectral_resolution:
    # hyperspectral:
    wave = np.arange(400, 701)
    # multispectral: - define sensor wavelengths
    # wave = np.array(sensor_wavelengths["OLCI-S3A"])

    # Generate figure to show NDI index relative to AVW
    fit4a = fit1 + 0.4
    fit4b = fit1 - 0.4

    ## Plot

    fig, ax = plt.subplots()

    ax.plot(avw_poly, fit1, '-k', linewidth=2)
    qwip_lines = []
    qwip_lines.append(ax.plot(avw_poly, fit4a, '--r', linewidth=2, label='QWIP ± 0.4')[0])
    qwip_lines.append(ax.plot(avw_poly, fit4b, '--r', linewidth=2)[0])

    ax.set_xlabel('AVW (nm)', fontsize=16)
    ax.set_ylabel(f'NDI ({wave[index_492]},{wave[index_665]})', fontsize=16)
    ax.set_ylim([-2.5, 2])
    ax.set_xlim([440, 630])

    for i in range(len(df_out)):
        ax.plot(df_out['avw'][i], df_out['ndi'][i], 'ok', markersize=1)

    return fig, ax, df_out, fit1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('df', type=str, help='Path to the input dataframe')
    parser.add_argument('spectral_resolution', type=str, help='Spectral resolution of the data: hyper or multi')
    parser.add_argument('wavelengths', type=str, help='Wavelengths of the multispectral sensor')
    parser.add_argument('sensor_coeffs', type=str, help='Sensor coefficients')
    args = parser.parse_args()
    
    # Load your df
    df = pd.read_csv('your_df.csv')
    fig, ax = plotQWIP(df, 'hyper')
    plt.show()