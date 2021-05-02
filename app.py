import matplotlib.pyplot as plt
import cv2
import os

from skimage.filters import gabor
from skimage import data, io,color
from skimage.color import rgb2gray
import numpy as np
import plotly.express as px
from scipy import ndimage
from sklearn.cluster import KMeans
import PIL
from PIL import Image
import streamlit as st

st.sidebar.title('Textile Detection')

st.title('Detect Defects on clothing items')

def f(a):
        # Read image.
        img = a.copy()
        
        # Convert to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Blur using 3 * 3 kernel.
        gray_blurred = cv2.blur(gray, (3, 3))
        
        # Apply Hough transform on the blurred image.
        detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                             param2 = 30, minRadius = 1, maxRadius = 40)
        
        # Draw circles that are detected.
        if detected_circles is not None:
        
                # Convert the circle parameters a, b and r to integers.
                detected_circles = np.uint16(np.around(detected_circles))
                
                for pt in detected_circles[0, :]:
                        a, b, r = pt[0], pt[1], pt[2]
                        
                        # Draw the circumference of the circle.
                        cv2.circle(img, (a, b), r, (0, 255, 0), 2)
                        
                        # Draw a small circle (of radius 1) to show the center.
                        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)
                      
                        break
        return img


image = st.sidebar.file_uploader("Upload an image", type = ['jpeg', 'jpg', 'png'])


if image is not None:
    
    option = st.selectbox('Choose an option',\
                          ('Original Image', 'Hough Transformation', 'Gray Scale Transformation','OpenCV'))
   
    
    image = Image.open(image)
    image = np.array(image.convert('RGB'))
    
    if option == "Original Image":
        st.subheader("Original Image")
        st.image(image, use_column_width = True)
    
    if option == "Hough Transformation": 
        st.subheader("Hough Transformation")
        b = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filt_real, filt_imag = gabor(b, frequency=0.05)
        gray = rgb2gray(image)
        
        gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
        for i in range(gray_r.shape[0]):
            if gray_r[i] > gray_r.mean():
                gray_r[i] = 3
            elif gray_r[i] > 0.5:
                gray_r[i] = 2
            elif gray_r[i] > 0.25:
                gray_r[i] = 1
            else:
                gray_r[i] = 0
        gray = gray_r.reshape(gray.shape[0],gray.shape[1])
        fig = px.imshow(gray)
        st.plotly_chart(fig)
    
        
    if option == "Gray Scale Transformation":
        b = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        st.subheader("Gray Scale Transformation")
        st.image(b, use_column_width = True)
    
    if option == "OpenCV":
        st.subheader("OpenCV Algo")
        img = f(image)
        st.image(img, use_column_width = True)
    
    
    
    
    
    
  
else:
    st.write("please upload an image in the formats shown above")