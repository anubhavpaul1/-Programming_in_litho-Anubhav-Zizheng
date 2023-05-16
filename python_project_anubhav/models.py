import numpy as np

import utilities as utls


class imaging_system:
    
    def __init__(self, sampling, na, wavlen, aberrations=[[0, 0, 0]], defocus=0):
        self.sampling = sampling
        self.na = na
        self.wavlen = wavlen
        self.aberrations = aberrations
        self.defocus = defocus
        
    def __call__(self, obj):
        
        img = self.coherent_imaging(obj)
        
        return img

        
    def calcu_pupil_sampling(self):
        '''
        This function computes the pupil plane sampling.
        '''

        norm_factor = self.na/self.wavlen

        pupil_sampling = utls.convert_ft_sampling(self.sampling, norm_factor)

        return pupil_sampling


    def calcu_pupil_sampling_grid(self):
        '''
        This function computes the pupil plane sampling grid.
        '''

        pupil_sampling = self.calcu_pupil_sampling()
        
        pupil_sampling_grid = utls.gen_sampling_array(pupil_sampling)

        return pupil_sampling_grid
    
    
    def print_sampling(self):
        '''
        This function computes and prints the object-image plane sampling and the pupil plane sampling.    
        '''
        
        pupil_sampling = self.calcu_pupil_sampling()

        nv = pupil_sampling[0][0]
        nu = pupil_sampling[0][1]
        dv = pupil_sampling[1][0]
        du = pupil_sampling[1][1]

        print('pupil plane sampling interval:')
        print('v: ' + "{:.4f}".format(dv))
        print('u: ' + "{:.4f}".format(du))
        print('pupil plane sampling field-of-view:')
        print('v: ' + "{:.2f}".format(nv*dv))
        print('u: ' + "{:.2f}".format(nu*du))

    
    def compute_wavefront(self):
        '''
        This function computes wavefront with aberrations as a weighted sum of zernike_polynomial. 
        '''
        
        pupil_sampling_grid = self.calcu_pupil_sampling_grid()

        [r, phi] = utls.cart2pol(*pupil_sampling_grid)

        wavefront = utls.compute_wavefront(r, phi, self.aberrations)
        
        return wavefront
    
    def compute_pupil(self):
        '''
        This function computes the pupil function with the wavefront computed with the aberrations.

        '''

        pupil_sampling_grid = self.calcu_pupil_sampling_grid()

        [r, phi] = utls.cart2pol(*pupil_sampling_grid)

        wavefront = compute_wavefront(r, phi, self.aberrations)
        
        # computing the defocus
        defocus = compute_defocus(self, r)

        # introduce the defocus to the pupil
        pupil = np.exp(2*np.pi*1j*wavefront) * np.exp(1j*defocus)
        pupil[r>1] = 0

        return pupil
    
    def compute_psf(self):
        '''
        This function computes the point-spread function with with the aberrations.

        '''

        pupil = self.compute_pupil()

        psf = utls.fft2(pupil)

        return psf
    
    def coherent_imaging(self, obj):
        '''
        This function computes the image of the object with a coherent imaging process described by Fourier optics.

        '''

        pupil = self.compute_pupil()

        img = np.abs(utls.ifft2(utls.fft2(obj)*pupil))**2

        return img


def compute_wavefront(r, phi, aberrations):
    '''
    This function computes wavefront with aberrations as a weighted sum of zernike_polynomial. 
    aberrations is a list in the following form:
    [[weight0, n0, m0],
     [weight1, n1, m1]
     ......
     [weightk, nk, mk]]
    '''
    
    def zernike_polynomial(r, phi, n, m):
        '''
        This function computes the zernike polynomial for radial order n and azimuthal order m.
        '''

        def radial_polynomial(r, n, m):

            V = 0
            for s in range(int((n-m)/2+1)):
                V += ((-1)**s * np.math.factorial(n-s))/np.math.factorial(s) *\
                    r**(n-2*s) / (np.math.factorial(int((n+m)/2-s))* np.math.factorial(int((n-m)/2-s)))

            return V

        if m>=0: # even
            F = radial_polynomial(r, n, np.abs(m)) * np.cos(np.abs(m)*phi)

        else:    # odd
            F = radial_polynomial(r, n, np.abs(m)) * np.sin(np.abs(m)*phi)


        return F

    wavefront = np.zeros_like(r)

    for l in range(len(aberrations)):

        weight = aberrations[l][0]

        n = aberrations[l][1]
        m = aberrations[l][2]

        wavefront += weight*zernike_polynomial(r, phi, n, m)

    wavefront[r>1] = 0

    return wavefront


def compute_defocus(self, r):
    
    norm_factor = np.pi*(self.na**2) /self.wavlen
    
    defocus_wavefront = norm_factor*self.defocus*(r**2) 
    
    return defocus_wavefront
