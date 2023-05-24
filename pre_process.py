import numpy as np
import configparser


config = configparser.ConfigParser()
config.read('config.ini')
def get_list_from_config(strr):
    req_list = [int(i) for i in strr.split(',')]
    return req_list

DIR = config.get('Paths', 'dir')
DIR_si10 = config.get('Paths', 'si10')
DIR_ssrd = config.get('Paths', 'ssrd')
DIR_u10 = config.get('Paths', 'u10')
DIR_v10 = config.get('Paths', 'v10')
DIR_r1000 = config.get('Paths', 'r1000')
DIR_z850 = config.get('Paths', 'z850')
DIR_u1000 = config.get('Paths', 'u1000')
DIR_v1000 = config.get('Paths', 'v1000')
DIR_era5 = config.get('Paths', 'processed_era5')

projection_dimensions = config.get('DataOptions','projection_dimensions')
projection_dimensions = get_list_from_config(projection_dimensions)
channels = int(config.get('DataOptions', 'channels'))

def load_from_numpy():
    """
    Loading Numpy arrays pf each Climate Variable

    """
    si10 = np.load(DIR_si10 + 'si10.npy' )
    ssrd = np.load(DIR_ssrd + 'ssrd.npy' )
    u10 = np.load(DIR_u10 + 'u10.npy')
    v10= np.load(DIR_v10 + 'v10.npy')
    r1000 = np.load(DIR_r1000 + 'r1000.npy')
    z850 = np.load(DIR_z850 + 'z850.npy')
    u1000 = np.load(DIR_u1000 + 'u1000.npy')
    v1000 = np.load(DIR_v1000 + 'v1000.npy')
    return si10, ssrd, u10, v10, r1000, z850, u1000, v1000

def combine_data(si10, ssrd, u10, v10, r1000, z850, u1000, v1000):
    """
    Stacking each Climate Variable across the individual channels.
    """
    X_final = np.zeros((channels, np.max(si10.shape), projection_dimensions[0], projection_dimensions[1]))
    X_final[0,] = si10
    X_final[1,] = ssrd
    X_final[2,] = u10
    X_final[3,] = v10
    X_final[4,] = r1000
    X_final[5,] = z850
    X_final[6,] = u1000
    X_final[7,] = v1000
    print("Shape ERA5 X: ", X_final.shape)
    print('------------')
    return X_final

def save_to_numpy(X_final):
    """
    Save the processed numpy data to respective folder.
    """
    np.save( DIR_era5 + 'X_final.npy',X_final)

    print("Data Saved!")

    return 0

if __name__ == "__main__":
    si10_r, ssrd_r, u10_r, v10_r, r1000_r, z850_r, u1000_r, v1000_r = load_from_numpy()
    print("Shape of raw ERA5 X data: ",si10_r.shape)
    assert si10_r.shape == r1000_r.shape == ( np.max(si10_r.shape), projection_dimensions[0], projection_dimensions[1])
    print('-------------')
    X_full_year = combine_data(si10_r, ssrd_r, u10_r, v10_r, r1000_r, z850_r, u1000_r, v1000_r)
    save_to_numpy(X_full_year)
    
