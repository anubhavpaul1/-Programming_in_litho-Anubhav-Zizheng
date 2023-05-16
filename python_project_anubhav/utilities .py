import numpy as np
from matplotlib import cm, colors


def fft2(f):
    
    g = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))/ np.sqrt(np.size(f))
    
    return g


def ifft2(g):
    
    f = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(g)))* np.sqrt(np.size(g))
    
    return f


def shifter(data, shift_coord):
    
    dy = int(shift_coord[0])
    dx = int(shift_coord[1])
    
    y0 = int(data.shape[0]/2)+1
    x0 = int(data.shape[1]/2)+1
    
    shifted_data = np.roll(data, (dy-y0, dx-x0), axis=(0,1))
    
    return shifted_data


def gen_sampling_array(sampling):
    
    def gen_sampling_array_1d(x_sampling):
        """
        Generates an array of coordinates with length nx and interval dd, \
        with the origin of the coordinates located at (nx/2)+1 for even nx and at (N+1)/2 for odd nx.

        Parameters:
        x_sampling (tuple): (nx, dx) a tuple containning the length and the interval of the array of           coordinates
        nx (int): The length of the array of coordinates.
        dx (int or float): The interval between the values in the array of coordinates.

        Returns:
        x (ndarray): An 1D array of coordinates.
        """
        
        nx = x_sampling[0]
        dx = x_sampling[1]
        
        if nx % 2 == 0:
            x0 = nx / 2+1
        else:
            x0 = (nx + 1) / 2
            
        x = np.arange(nx) + 1 - x0
        x *= dx
        
        return x

    ny = sampling[0][0]
    nx = sampling[0][1]
    dy = sampling[1][0]
    dx = sampling[1][1]

    y = gen_sampling_array_1d((ny, dy))
    x = gen_sampling_array_1d((nx, dx))
    
    xx, yy = np.meshgrid(x, y)

    return [yy, xx]


def convert_ft_sampling(sampling, norm_factor):
    
    def convert_ft_sampling_1d(x_sampling, x_norm_factor):
        """
        Converts the sampling in the spatial domain to the sampling in the spatial frequency domain.

        Parameters:
        x_sampling (tuple): (nx, dx) A tuple containing the sampling number and the sampling interval
        nx (int): The sampling number in the spatial domain.
        dx (int or float): The sampling interval in the spatial domain.

        Returns:
        u_sampling (tuple): (nu, du) A tuple containing the sampling number and the sampling interval         in the spatial frequency domain.
        """
    
        nx = x_sampling[0]
        dx = x_sampling[1]

        nu = nx
        du = 1 / (nx * dx) / x_norm_factor
        u_sampling = (nu, du)

        return(u_sampling)
    
    ny = sampling[0][0]
    nx = sampling[0][1]
    dy = sampling[1][0]
    dx = sampling[1][1]

    (nv, dv) = convert_ft_sampling_1d((ny, dy), norm_factor)
    (nu, du) = convert_ft_sampling_1d((nx, dx), norm_factor)
    
    return ((nv, nu), (dv, du))


def cart2pol(yy, xx):
    
    r = np.sqrt(yy**2 + xx**2)
    phi = np.arctan2(yy, xx)
    
    return [r, phi]


def exponential(psi):
    
    return np.exp(1j*2*np.pi*psi)


def generate_rectangle(x, w, c):
    """
    Generates a 1D rectangular function with width w and center c, given an array of coordinates x.

    Parameters:
    x (ndarray): An 1D array of coordinates.
    w (int or float): The width of the rectangle.
    c (int or float): The center of the rectangle.

    Returns:
    rectangle (ndarray): An 1D array containing the rectangular function.
    """

    l_lim = c - w / 2
    r_lim = c + w / 2

    rectangle = np.zeros(x.shape)

    rectangle[(x >= l_lim) & (x <= r_lim)] = 1

    return rectangle


