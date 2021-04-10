# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:40:43 2021

@author: maya_
"""

def showImage(img):

    # importing Image class from PIL package 
    from PIL import Image 
  
    # creating a object 
    im = Image.open(img) 
  
    im.show()