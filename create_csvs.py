import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, img_as_float
from skimage.measure import find_contours, label, regionprops, regionprops_table
import cv2
from scipy import ndimage
import pandas as pd


def find_ellipses_in_image(image, folder_1, folder_2):
    '''
    From the artificially generated image, we are able to access the locations and
    orientations of the ellipses. These are all added to a csv file which is saved
    in the folder locations specified as strings by the user.
    '''
    image = io.imread(path, as_gray=True)
    height, width = image.shape
    area = height*width

    ret_white, thresh_white = cv2.threshold(image, 200, 0, cv2.THRESH_TOZERO)
    ret_white, thresh_black = cv2.threshold(image, 200, 255, cv2.THRESH_OTSU)

    white_mask = thresh_white == 255
    black_mask = thresh_black != 255

    s = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    white_label, white_black = ndimage.label(white_mask, structure=s)
    black_label, num_black = ndimage.label(black_mask, structure=s)

    white_clusters = regionprops(white_label)
    black_clusters = regionprops(black_label)

    #print(white_clusters)
    black_area = 0
    orientations_black = []
    black_centroids = []
    black_axis_lengths = []
    locations1 = pd.DataFrame(columns=['centroidx', 'centroidy', 'minor_axis_length', 'major_axis_length', 'orientation'])
    for k in range(np.size(black_clusters)):
        black_area += black_clusters[k].area
        orientations_black.append(black_clusters[k].orientation)
        black_centroids.append(black_clusters[k].centroid)
        black_axis_lengths.append([black_clusters[k].minor_axis_length, black_clusters[k].major_axis_length])
        dataframe_to_add = pd.DataFrame([[black_centroids[k][0], black_centroids[k][1], black_axis_lengths[k][0], black_axis_lengths[k][1], orientations_black[k]]], columns=['centroidx', 'centroidy', 'minor_axis_length', 'major_axis_length', 'orientation'])
        locations1 = pd.concat([locations1, dataframe_to_add], ignore_index=True)

    white_area = 0
    orientations_white = []
    white_centroids = []
    white_axis_lengths = []
    locations2 = pd.DataFrame(columns=['centroidx', 'centroidy', 'minor_axis_length', 'major_axis_length', 'orientation'])
    for j in range(np.size(white_clusters)):
        white_area += white_clusters[j].area
        orientations_white.append(white_clusters[j].orientation)
        white_centroids.append(white_clusters[j].centroid)
        white_axis_lengths.append([white_clusters[j].minor_axis_length, white_clusters[j].major_axis_length])
        dataframe_to_add = pd.DataFrame([[white_centroids[j][0], white_centroids[j][1], white_axis_lengths[j][0], white_axis_lengths[j][1], orientations_white[j]]], columns=['centroidx', 'centroidy', 'minor_axis_length', 'major_axis_length', 'orientation'])
        locations2 = pd.concat([locations2, dataframe_to_add], ignore_index=True)


    black_volume_fraction = black_area/area
    white_volume_fraction = white_area/area
    avg_orientations = np.mean(orientations_black)


    locations1.to_csv(folder_1 '.csv')
    locations2.to_csv(folder_2 + '.csv')
return
