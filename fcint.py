import numpy as np
import math

def fcint(dzdx,dzdy,pz):
    dzdx = np.divide(-dzdx,(pz+np.spacing(1)));
    dzdy = np.divide(-dzdy,(pz+np.spacing(1)));
    
    dzdx[np.isnan(dzdx)]=0
    dzdy[np.isnan(dzdy)]=0
    
    [rows,cols] = np.shape(dzdx);

    [wx, wy] = np.meshgrid((range(1,cols+1)-(np.fix(cols/2)+1))/(cols-np.mod(cols,2)), (range(1,rows+1)-(np.fix(rows/2)+1))/(rows-np.mod(rows,2)));

    wx = np.fft.ifftshift(wx); 
    wy = np.fft.ifftshift(wy);
    
    DZDX = np.fft.fft2(dzdx)   
    DZDY = np.fft.fft2(dzdy);

    wxj=-1j*wx
    wyj=1j*wy
    dividend=(np.multiply(wxj,DZDX) -np.multiply(wyj, DZDY))
    divisor=(wx**2 + wy**2 + np.spacing(1))
   
    Z=np.divide(dividend,divisor)

    z = np.fft.ifft2(Z); 
    z=z.real
    z = z - min(z.flatten(1));
    z = z/math.pi/2;
    
    return z
