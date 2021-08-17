# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 15:03:59 2020
Author: Oussama Chaib

Combustion image processing code for OH* chemiluminescence
Compatible with Lightfield (.spe)
"""

from pylab import*
import spe2py as spe
import spe_loader as sl
import cv2
import os
import abel as ab

close('all')

tic=time.time()

# Image and save path

path = "/Users/oussamachaib/Desktop/"
folder = "Testing Python Codes"+"/"
file = "test"
ext = ".spe"


# Number of frames per partition
N_img = 20

# Cropping limits
# Top left
x1 = 0
y1 = 0
# Bottom right
x2 = 1024    
y2 = 1024

# Initialize matrices

sum_img = zeros((y2-y1,x2-x1))
avg = zeros((y2-y1,x2-x1))
img_backup=zeros((y2-y1,x2-x1))




# Nomenclature
d1=6
d2=16
alpha=0
S1=1.42
# Variables
P=70
J=3
S_2=0.5

# Read partition, compute average, back up instantaneous frame

print("Loading partition data...")


file_object=sl.load_from_files([path+folder+file+ext])
for j in range(0,N_img):
    print('Reading frame '+str(j+1)+'/'+str(N_img))
    frame_data = file_object.data[j]
    img0 = frame_data[0]
    img0 = img0[y1:y2,x1:x2]
    sum_img = sum_img + img0
img_backup[:,:]=img0

print("Calculating average frame...")

avg[:,:] = sum_img/N_img             

# Flame edge detection by Otsu thresholding

print("Detecting flame edge...")

# Convert image to 8-bit
img=255*avg[:,:]/avg[:,:].max() 
img=img.astype(np.uint8)

# Filter by Gaussian kernel
blur = cv2.GaussianBlur(img,(11,11),0)
# Apply Otsu threshold
_, img2 = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
# Select flame edge by Canny contour
edges = cv2.Canny(img2,0,0)
# Thicken flame contour by dilatation
kernel = np.ones((4,4),np.uint8)
edges = cv2.dilate(edges,kernel,iterations = 2) 
# Combine average image and Otsu contour
added_image = cv2.addWeighted(img,0.6,edges,1,100)  

print('Computing Abel deconvoluted image...')

# Abel inversion

# X-coordinate of vertical burner axis
c = 337 

# Filter by Gaussian kernel
blur = cv2.GaussianBlur(img.T,(5,5),0)
# Center filtered image
original= blur[:,0:c+c]
# Compute Abel inverted image
inverse_abel = ab.transform.Transform(original, direction='inverse', method='onion_peeling').transform

# Data vizualization

print('Plotting data...')

# Average frame and contour
subplot(121)
imshow(added_image[0:c+c,:],cmap='hot')
title('Average OH* CL + Contour')
# Abel inverted image
subplot(122)
imshow(inverse_abel.T,cmap='hot',interpolation='nearest')
title('Abel deconvoluted')
# Adjust Abel colorbar for better visibility
clim(vmin=0,vmax=2)
# Displaty nomenclature
myTitle=r"$d_1$ = "+str(d1)+r" mm; $d_2$ = "+str(d2)+r" mm; $\alpha$ = "+str(alpha)+"°\n"+r"P = "+str(P)+r" kW; J = "+str(J)+r" ; $S_1$ = "+str(S1)+r" ; $S_2$ = "+str(S_2)
suptitle(myTitle, wrap=True,fontsize=15)
plt.tight_layout()

print('\n[*] Data plotted successfully :)')
print('[*] Elapsed time (in seconds):', round(time.time()-tic,2))
