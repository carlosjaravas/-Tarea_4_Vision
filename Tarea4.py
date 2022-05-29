#Importar las librerías por utilizar
import os
import pandas as pd
from skimage import io
from scipy import ndimage
from skimage import exposure
from skimage.morphology import disk
from skimage.morphology import closing
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops


#Funcion del preprocesado
def prepro(img, thresh=0.3, sigma = 3):
    #Se realiza la ecualizacion adaptativa
    EQUA = exposure.equalize_adapthist(img, clip_limit=0.03)
    f = EQUA
    #Se aplica un filtro gaussiano para eliminar el ruido de la imagen ecualizada
    blurred_f = ndimage.gaussian_filter(f, sigma)
    #Se binariza la imagen ecualizada y binarizada
    binary_blurred_f = blurred_f > thresh
    binary_blurred_f = ~binary_blurred_f
    #Se unen figuras distanciadas por una distancia maxima de 20 pixeles
    footprint = disk(10)
    limp = closing(binary_blurred_f,footprint)*255
    #Se elimina el ruido de los bordes de la imagen
    no_borders = clear_border(limp)
    return no_borders


#Funcion de clasificacion
def clasificacion(regions):
    #Se crean los contadores de objetos
    clavos = 0
    arandelas = 0
    espanders = 0
    prensas = 0
    limones = 0
    #Se determinan las caracteriticas de cada region segmentada
    props = regionprops(regions)
    for prop in props:
        #Se discriminan areas pequeñas pues son no deseadas
        if prop.area > 700:
            #Se obtiene el ancho de la region
            h = prop.axis_minor_length
            #Se obtiene el largo de la region
            l = prop.axis_major_length
            #Se calcula la relacion entre el area del region y el area 
            #del rectanguloque con la misma orientacion que la region que
            #la encierra
            area_ratio = prop.area/(h*l)
            #Se calcula la relacion de aspecto
            aspect_ratio = l/h
            #Se obtiene el area de las regiones con los agujeros
            #rellenos
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
    return clavos, arandelas, prensas, espanders, limones


#Se crea la funcion principal que ejecuta el codigo para un conjunto 
#de imagenes
def main():
    if os.path.exists("report_file.xlsx"):
        os.remove("report_file.xlsx")
    #Se obtiene el path del .py y de los archivos relevantes
    path = os.getcwd()
    folder = 'imagenes_por_analizar'
    folder_path = os.path.join(path, folder)
    files = os.listdir(folder_path)
    #Se crea una tabla de una columna con la lista de archivos
    files_df = pd.DataFrame(files, columns = ["File"])
    #Se crea la lista de resultados
    results = []
    #Se realiza el preprocesado y clasificacion para cada uno de
    #los archivos seleccionados
    for file in files:
        #Se carga la imagen
        file_path = os.path.join(folder_path, file)
        foto = io.imread(file_path)
        #Preprocesado
        prepro_img = prepro(foto)
        #Segmentacion de regiones
        regions = label(prepro_img,connectivity= None)
        #Clasificacion de regiones
        file_res = clasificacion(regions)
        results.append(file_res)
    #Se crea la tabla con los resultados por imagen
    res_df = pd.DataFrame(results, columns = (["Clavos", "Arandelas", "Prensas", "Espanders", "Limones"]))
    finalDf = pd.concat([files_df, res_df], axis = 1)
    finalDf.to_excel("report_file.xlsx")


if __name__ == "__main__": main()