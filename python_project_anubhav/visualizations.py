# import cv2
import time
import numpy as np
from matplotlib import cm, colors
import matplotlib.pyplot as plt
from IPython.display import display, Image, clear_output


def imshow(data, engine='matploblib', 
           fig=None, ax=None, cmap=None, title=None):
    """
    Display data (in the form of a matrix, both real and complex) as an image using a colormap in a Jupyter Notebook.
    """

    if not np.iscomplexobj(data):
        if cmap is None:
            cmap = 'turbo'
        data = plot_real_data(data, cmap)
    else:
        if cmap is None:
            cmap = 'hsv'
        data = plot_complex_data(data, cmap)
    
    if engine=='matploblib':
        
        if (fig is None) and (ax is None):
            fig, ax = plt.subplots()
        
        ax.clear()
        ax.imshow(data)
        ax.axis('off')
        ax.set_title(title)
        # needed when show image in jupyter notebook
        display(fig)
        # needed when show image in a loop
        clear_output(wait=True)
        time.sleep(1e-6)

        return fig, ax
    
    elif engine=='opencv':
        
        # normalize data
        normalized_data = (data * 255).astype(np.uint8)

        # convert color from bgr2rgb
        colored_data = cv2.cvtColor(normalized_data, cv2.COLOR_BGR2RGB)

        # encode the colored data in PNG format
        is_success, buffer = cv2.imencode('.png', colored_data)

        # display the colored data in the jupyter notebook with IPython
        if is_success:
            # needed when show image in jupyter notebook
            display(Image(data=buffer.tobytes()))
            # needed when show image in a loop
            clear_output(wait=True)
            time.sleep(1e-6)
        else:
            raise ValueError("failed to encode the colored data in PNG format")
    
    else:
        raise ValueError("unkown engine '" + engine + "' cannot be identified")

        
def plot_real_data(plot_data, cmap='hsv'):
    
    amplitude = (np.abs(plot_data))
    amplitude -= np.amin(amplitude)
    amplitude /= np.amax(amplitude)

    cmap = cm.get_cmap(cmap)
    img_rgb = cmap(amplitude)
    
    return img_rgb


def plot_complex_data(plot_data, cmap='hsv'):
    
    amplitude = (np.abs(plot_data))
    amplitude -= np.amin(amplitude)
    amplitude /= np.amax(amplitude)

    clim_pha = [0, 2*np.pi]
    phase = np.angle(plot_data)
    phase = phase + np.pi
    phase[phase > clim_pha[1]] = clim_pha[1]
    phase[phase < clim_pha[0]] = clim_pha[0]
    phase = phase / (clim_pha[1]-clim_pha[0])

    cmap = cm.get_cmap(cmap)
    img_rgb = cmap(phase)
    img_rgb = img_rgb[:, :, [0, 1, 2]]
    img_hsv = colors.rgb_to_hsv(img_rgb)
    img_hsv[:, :, 2] = img_hsv[:, :, 2] * amplitude
    img_rgb = colors.hsv_to_rgb(img_hsv)
    
    return img_rgb
