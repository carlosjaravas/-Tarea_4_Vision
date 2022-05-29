#Importar las librerías por utilizar
import os
import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import math
import csv
from skimage import io
from skimage.transform import resize, probabilistic_hough_line,hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage.color import rgb2gray
from skimage.draw import circle_perimeter
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import (erosion, dilation, opening, closing)
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage import data, img_as_float
from skimage import exposure
from scipy import ndimage


def prepro(img, thresh=0.3, sigma = 3):
    EQUA = exposure.equalize_adapthist(img, clip_limit=0.03)
    f = EQUA
    blurred_f = ndimage.gaussian_filter(f, sigma)
    binary_blurred_f = blurred_f > thresh
    binary_blurred_f = ~binary_blurred_f
    footprint = disk(10)
    limp = closing(binary_blurred_f,footprint)*255
    no_borders = clear_border(limp)
    return no_borders

def clasificacion(regions):
    clavos = 0
    arandelas = 0
    espanders = 0
    prensas = 0
    limones = 0
    error = 0
    props = regionprops(regions)
    for prop in props:
        if prop.area > 700:
            h = prop.axis_minor_length
            l = prop.axis_major_length
            area_ratio = prop.area/(h*l)
            aspect_ratio = l/h
            area = prop.area_filled
            #identifica limones
            if (area >= 8000):
                limones +=1
            #identifica clavos
            elif (area >= 1200 and area < 3100) and (aspect_ratio >= 2.52 and aspect_ratio < 11) and (area_ratio >= 0.35 and area_ratio < 0.56):
                clavos +=1
            #identifica arandelas
            elif (area >= 800 and area < 2000) and (aspect_ratio >= 1.5 and aspect_ratio < 1.95) and (area_ratio >= 0.77 and area_ratio < 0.79):
                arandelas +=1
            #identifica prensas
            elif (area >= 2500 and area < 5200) and (aspect_ratio >= 1.35 and aspect_ratio < 3.39) and (area_ratio >= 0.5 and area_ratio < 0.66):
                prensas +=1
            #identifica espanders
            elif (area >= 1400 and area < 3700) and (aspect_ratio >= 2.1 and aspect_ratio < 7.1) and (area_ratio >= 0.73 and area_ratio < 0.77):
                espanders +=1
            else:
                error +=1
    return clavos, arandelas, prensas, espanders, limones, error

def main():
    if os.path.exists("report_file.csv"):
        os.remove("report_file.csv")
    #report_file = csv.open("report_file.csv", "x")
    with open("report_file.csv", "w", newline = "", encoding='utf-8') as report_file:
        report_writer = csv.writer(report_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        report_writer.writerow(["File", "Clavos", "Arandelas", "Prensas", "Espanders", "Limones", "No identificados"])
        path = os.getcwd()
        folder = 'MUESTRAS'
        folder_path = os.path.join(path, folder)
        files = os.listdir(folder_path)
        for file in files:
            file_path = os.path.join(folder_path, file)
            foto = io.imread(file_path)
            prepro_img = prepro(foto)
            regions = label(prepro_img,connectivity= None)
            file_res = clasificacion(regions)
            report_writer.writerow([file, file_res[0], file_res[1], file_res[2], file_res[3], file_res[4], file_res[5]])
            #res_report =file + str(file_res) + '\n'
            #report_file.write(res_report)
        #report_file.close()

if __name__ == "__main__": main()