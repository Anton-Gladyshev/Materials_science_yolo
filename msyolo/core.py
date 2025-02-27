import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import scipy
import cv2
from scipy.interpolate import interp2d
import scipy.interpolate as spip
import matplotlib.image as mpimg
import csv
from scipy.optimize import curve_fit


from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from matplotlib import rcParams
import matplotlib.patheffects as PathEffects
import matplotlib.animation as animation
from IPython.display import HTML
import sys
from tqdm import tqdm
from matplotlib.colors import LogNorm




def gamma_correction(image, gamma=1.0):
    minval=np.min(image)
    image=image-minval
    maxval=np.max(image)
    image=image/maxval
    corrected_array =image ** gamma
    image=image*maxval+minval
    return corrected_array
    
    
def upsample(array, pad):
    array=np.fft.fftshift(np.fft.fft2(array))
    array2=np.zeros((array.shape[0]+2*pad, array.shape[1]+2*pad), dtype=np.complex128)
    array2[pad:-pad,pad:-pad]=array
    return np.real(np.fft.ifft2(np.fft.ifftshift(array)))



def find_positons_yolo(array, model, plot=True, img=None, yolo_dict={}):
    if img is None:
        img=array-np.min(array)
        img/=np.max(img)
        img=np.stack([img, img, img])
        img=np.swapaxes(img, 0,1)
        img=np.swapaxes(img, 1,2)
        img=(255*img/np.max(img)).astype(np.uint8)
        model = YOLO(model)
        results = model(img, **yolo_dict)
        xywh=results[0].boxes.xywh.cpu().numpy()
        N_atoms=xywh.shape[0]
        if plot:
            fig,ax=plt.subplots(figsize=(5,5))
            ax.imshow(results[0].orig_img)
            for this_xy in xywh:
                x,y,w,h=this_xy
                r=0.5*(w**2+h**2)**0.5
                circ=Circle([x,y], r, fill=None, alpha=1, edgecolor="blue")
               
                ax.add_patch(circ)
            ax.axis("off")
            plt.show()
        return xywh

def refine_from_xywh(array, xywh, R=None, pos_type="com", thresh=0):
    x=np.arange(array.shape[1])
    y=np.arange(array.shape[0])
    x,y=np.meshgrid(x,y)
    coms=np.empty((xywh.shape[0], 2))

    this_array=np.copy(array[:,:])
    for i_atoms in range(xywh.shape[0]):
        this_x, this_y,this_w,this_h=xywh[i_atoms]
        this_w, this_h=this_w, this_h
        if R is None:
            r=((x-this_x)**2+(y-this_y)**2)<=0.25*(this_w**2+this_h**2)#+1.5
        else:
            r=((x-this_x)**2+(y-this_y)**2)<=R**2
        this_array_crop=np.copy(this_array[r])
        this_x=x[r]
        this_y=y[r]
        this_array_crop-=np.min(this_array)
        this_array_crop=(this_array_crop>thresh*np.max(this_array_crop))
        if pos_type=="com":
            coms[i_atoms, 0]=np.average(this_x,weights=this_array_crop)
            coms[i_atoms, 1]=np.average(this_y,weights=this_array_crop)
        else:
            coms[i_atoms, 0]=this_x[np.argmax(this_array_crop)]
            coms[i_atoms, 1]=this_y[np.argmax(this_array_crop)]
    return coms

def fit_vector_field(x, y, u, v, Xi,Yi,method='nearest'):
    x,y,u,v=x.flatten(),y.flatten(),u.flatten(),v.flatten()
    Ui = griddata((x, y), u, (Xi, Yi), method=method, fill_value=0)
    Vi = griddata((x, y), v, (Xi, Yi), method=method, fill_value=0)
    return Ui, Vi


def apply_correction(array, model_path, plot=True, yolo_dict={}, R=None, pos_type="com", method="nearest", order=1, thresh=0):
    phase=get_mean_phase_image(array)
    xywh=find_positons_yolo(phase, model_path,yolo_dict=yolo_dict, plot=plot)
    coms=refine_from_xywh(np.angle(array), xywh, R=R, pos_type=pos_type, thresh=thresh)
    corrected=correct_all_states(array, coms, plot=plot, method=method, order=order)
    return corrected
    
    
def radial_average(im_full, x,y, r_bins, r_max):
    im_fulls=im_full.shape
    if x<(im_fulls[1]-r_max) and x>r_max and y>r_max and y<(im_fulls[0]-r_max):
        x=int(x)
        y=int(y)
        
        x_grid,y_grid=np.arange(-r_max,r_max+1,1), np.arange(-r_max,r_max+1,1)
        im=np.copy(im_full[y-r_max:r_max+1+y, x-r_max:x+1+r_max])
        im-=np.min(im)
        x_grid,y_grid=np.meshgrid(x_grid, y_grid, indexing="xy")
        r=(x_grid**2+y_grid**2)**0.5
        unique_distances=np.arange(r_bins, r_max, r_bins)
        radial_avg=np.zeros_like(unique_distances)
        for iii in range(len(unique_distances)):
            distance=unique_distances[iii]
            radial_mask = (r<=distance+r_bins) *(r>distance)
            radial_avg[iii] = np.mean(im[radial_mask])

        return radial_avg, unique_distances
    else:
        return None, None



def get_xywh_from_numpy(array, model_path, yolo_dict={'conf':0.5,
          'iou': 0.1,
           'device': 'cpu',
           'max_det': 800,
           'retina_masks': False}, plot=True):
    xywh=find_positons_yolo(array, model_path,yolo_dict=yolo_dict, plot=plot)
    return xywh
    
    

def get_radial_profiles_for_atoms(array, xywh, pixel_size, r_bins, r_max):
    profiles=[]
    for i in tqdm(range(xywh.shape[0])):
        p,u=radial_average(array, xywh[i,0],xywh[i,1], r_bins, r_max)
        if not(p is None):
            profiles.append(p)
            unique_distances=u
    profiles=np.array(profiles)
    radavgy=np.mean(profiles[:,:],0)
    unique_distances*=pixel_size
    return profiles, unique_distances



def plot_xywh_on_image(array, xywh, R=None):
    fig,ax=plt.subplots(figsize=(5,5))
    ax.imshow(array, cmap="gray")
    for this_xy in xywh:
        x,y=this_xy[:2]
        if R is None:
            r=0.5*((this_xy[:2]**2).sum())**0.5
        else:
            r=R
        circ=Circle([x,y], r, fill=None, alpha=1, edgecolor="blue")
        ax.add_patch(circ)
    ax.axis("off")
    plt.show()
    return fig
