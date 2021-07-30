import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import io
from skimage import feature
import random
from scipy.spatial import Delaunay
import cv2
from matplotlib.patches import Ellipse
from shapely.geometry.polygon import LinearRing
import collections


def pointInEllipse(x,y,xp,yp,d,D,angle):
    '''
    A function that can recognise whether a point with x and y co-ordinates is in
    a certain ellipse

    Parameters
    --------

    x: The x position of the point

    y: The y position of the point

    xp: The x position of the centroid of the ellipse

    yp: The y position of the centroid of the ellipse

    d: The minor axis length of the ellipse

    D: The major axis length of the ellipse

    angle: The orientation of the ellipse, how it is offset from the origin

    --------
    Returns
    --------
    A boolean value as to whether the point is in the ellipse
    '''

    # Find the angles that can be used for the trigonometry
    cosa = math.cos(angle)
    sina = math.sin(angle)
    # Using the major and minor axis lengths for the algorithm
    dd = d/2*d/2
    DD = D/2*D/2
    # Implementing the rest of the algorithm for if a point is enclosed inside
    # an ellipse
    a = math.pow(cosa*(xp-x)+sina*(yp-y), 2)
    b = math.pow(sina*(xp-x)-cosa*(yp-y), 2)
    # If the manor/minor axis lengths are zero then return False
    if dd == 0 or DD == 0:
        return False
    ellipse=(a/dd)+(b/DD)
    if ellipse <= 1:
        return True
    else:
        return False


def point_in_ellipse(x, y, x0, y0, a, b, theta):
    '''
    A function that uses the other pointInEllipse function to see if the point is
    in the ellipse. Works the exact same as this function so could not be used, however
    I still use the function
    '''
    truth_value = False
    # If the above function is True then return True
    if pointInEllipse(x, y, x0, y0, a, b, theta):
        truth_value = True
    return truth_value


def polar_form(r, theta):
    '''
    A function that can find the spatial vector change given a change in r and a
    change in theta.

    Parameters
    --------

    r: The modulus of the vector change

    theta: The angle of the vector change

    --------
    Returns
    --------
    A vector caused by a change in r and a change in theta
    '''
    return [r*math.cos(theta), r*math.sin(theta)]


def get_all_line_points(r, height, width):
    '''
    A function that can find a group of line points based on the vector between
    the first and last points in the sequence.

    Parameters
    --------

    r: The distance that the first and last points are separated by

    height: The height of the image so a random point in that range can be found

    width: The width of the image so a random point in that range can be found

    --------
    Returns
    --------
    A vector of evenly spaced points from the first generated point to another
    point separated by the vector r.
    '''
    points = []
    # Find the initial point that can then be iterated on from
    point = np.array([random.uniform(0, width-r-1), random.uniform(0, height-r-1)])
    points.append(point)
    angle = random.uniform(0, 2*math.pi)
    # Find how many points you want along the line
    increments = r/10
    for i in range(10):
        # Add the polar form using the r and angle
        addition = np.array(polar_form(increments, angle))
        point = point + addition
        points.append(point)
    return np.asarray(points)


def k_means(image, num_clusters):
    '''
    A function that can perform the k-means algorithm on an image to determine
    how many different phases exist in the image

    Parameters
    --------

    image: The array that represents the values of all the pixels in the image

    num_clusters: The number of clusters that you want the image to be split into

    --------
    Returns
    --------

    An array of which cluster each pixel belongs to
    '''
    clusters = np.zeros(np.shape(image))
    k = num_clusters
    means = []
    for i in range(k):
        # add k number of random starting means to an array
        means.append(random.uniform(np.min(image), np.max(image)))
    while True:
        # initialise a clusters matrix to put the number of the cluster
        prev_clusters = np.copy(clusters)
        # loop through all the values of the image
        for i in range(np.shape(image)[0]):
            for j in range(np.shape(image)[1]):
                # initialise what the lowest difference between the value and the means is
                lowest_diff = np.inf
                # calculate the difference in all of the means, if it is the lowest then add it
                for mean in means:
                    # if the difference between the difference is low enough make this the lowest difference
                    if abs(mean - image[i][j]) < lowest_diff:
                        lowest_diff = abs(mean - image[i][j])
                        # return the index of the lowest difference
                        cluster = means.index(mean)
                clusters[i][j] = cluster

        # Now we know how to find out which cluster each value is in, we can recalculate the means
        # But first we check if the algorithm has converged
        converged = (clusters == prev_clusters).all()
        if converged:
            break

        # Now lets sort out the means
        # loop through the cluster values
        for k_value in range(k):
            ks = []
            # loop through every single clusters value
            for i in range(np.shape(clusters)[0]):
                for j in range(np.shape(clusters)[1]):
                    # if this cluster value is the certain k, then add it to an array
                    if clusters[i][j] == k_value:
                        ks.append(clusters[i][j])
            means[k_value] = np.mean(ks)
    return clusters


def ellipse_polyline(ellipses, n=100):
    '''
    A function that can create a line representing all the points in the ellipse,
    across a range of angles

    Parameters
    --------

    ellipses: The group of ellipses represented by the centroid, the minor/major
    axis length, and the orientation

    n: The number of values to have the linspace for

    --------
    Returns
    --------

    An array of all the polylines that represent the ellipses
    '''
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    result = []
    for x0, y0, a, b, angle in ellipses:
        angle = np.deg2rad(angle)
        sa = np.sin(angle)
        ca = np.cos(angle)
        p = np.empty((n, 2))
        p[:, 0] = x0 + a * ca * ct - b * sa * st
        p[:, 1] = y0 + a * sa * ct + b * ca * st
        result.append(p)
    return result


def intersections(a, b):
    '''
    A function that finds the intersection points of two ellipses

    Parameters
    --------

    a: The ellipse polyline for the first ellipse

    b: The ellipse polyline for the second ellipse

    --------
    Returns
    --------

    The intersection point of the two ellipse polylines
    '''
    ea = LinearRing(a)
    eb = LinearRing(b)
    mp = ea.intersection(eb)
    return mp


def neighbouring_pixels(pixel, edges):
    '''
    A function that can find all of the neighbours to a pixel that are a certain
    value in an image array

    Parameters
    --------

    pixel: The x and y value of the pixel that you are finding the neighbours for

    edges: The array of the image once the canny edge detector

    --------
    Returns
    --------

    An array of all the neighbours that share the same phase as the initial pixel
    '''
    x, y = pixel
    neighbours = []
    # Access the points surrounding the pixel. This is just incase there is a
    # small gap between two pixels of the same phase.
    xs = [x-2, x-1, x, x+1, x+2]
    ys = [y-2, y-1, y, y+1, y+2]
    for x_value in xs:
        for y_value in ys:
            if x_value < edges.shape[0] and y_value < edges.shape[1]:
                if edges[x_value][y_value] and (x_value, y_value) not in neighbours:
                    neighbours.append((x_value, y_value))
    return neighbours


def scan_through_from_pixel(pixel, edges):
    '''
    A function that can scan through from an initial pixel, and keep iteratively
    finding neighbours of points so that a group of points can be recognised

    Parameters
    --------

    pixel: The x and y value of the pixel that you are finding the neighbours for

    edges: The array of the image once the canny edge detector

    --------
    Returns
    --------

    An array of all the grouped points based on their neighbours.
    '''
    grouped = []
    # Find the neighbours of a pixel.
    neighbours = neighbouring_pixels(pixel, edges)
    for neighbour in neighbours:
        grouped.append(neighbour)
    # Once we have the first neighbours, we iteratively search through the rest
    # of the points to find a full group
    for pixel in grouped:
        more_neighbours = neighbouring_pixels(pixel, edges)
        for neighbour in more_neighbours:
            if neighbour not in grouped:
                grouped.append(neighbour)
    return grouped


def find_point_to_be_grouped(edge, grouped_array):
    '''
    A function that can find an initial point in the image to scan through from
    and find its neighbours

    Parameters
    --------

    edge: The array after being applied to the canny operator

    grouped_array: The array of grouped points. This is used to check that the
    point that has been found is not already in a cluster

    --------
    Returns
    --------

    A new point that can be looped through from to find a set of points.
    '''
    truth = False
    while truth == False:
        point = [random.randint(0, edge.shape[0]-1), random.randint(0, edge.shape[1]-1)]
        # Providing that the point is not already in a group, and it is part of an
        # edge then this point can be the start of the iteration process to find a
        # group.
        if point not in grouped_array and edge[point[0], point[1]]:
            truth = True
    return point


def line_between_points(point_1, point_2, colour):
    '''
    A function that can draw a line between two points when representing the
    two-point cluster function or the lineal-path function. Either red or green
    dependent on the boolean value of the function.

    Parameters
    --------

    point_1: The point that the line is to be drawn from

    point_2: The point that the line is to be drawn to

    colour: Either 'r' or 'g' which represent the boolean value of the two-point
    function

    --------
    Returns
    --------
    None
    '''
    xs = np.array([point_1[0], point_2[0]])
    ys = np.array([point_1[1], point_2[1]])
    plt.plot(xs, ys, c=colour)
    return


def proportion_of_array(array):
    '''
    A function that can calculate the proportion of the array that is a 1. From
    this, we can find out which of the array belong to the matrix, fibers and voids

    Parameters
    --------

    array: The array for which the proportion will be found

    --------
    Returns
    --------

    A fraction of the array that is a 1
    '''
    count = np.sum(np.count_nonzero(array == 1, axis=1))
    return count/np.size(array)


def find_overlapping_ellipses(ellipses):
    '''
    A function that can find the ellipses that overlap in the group of ellipses
    found for the image

    Parameters
    --------

    ellipses: The array of the ellipses in the image

    --------
    Returns
    --------

    An array of pairs of ellipses that overlap. The index of the ellipse is included
    in each pair
    '''
    overlapping = []
    for p in range(np.shape(ellipses)[0]):
        for q in range(np.shape(ellipses)[0]):
            # Providing that we are not searching for intersections of an ellipse
            # with itself
            if p != q:
                group = [ellipses[p], ellipses[q]]
                a, b = ellipse_polyline(group)
                # These next steps are to prevent an error caused by when the ellipse
                # polyline has the same values in it
                if np.size(np.unique(np.asarray(a)[:, 1])) == 1 or np.size(np.unique(np.asarray(a)[:, 0])) == 1:
                    intersection_points = []
                elif np.size(np.unique(np.asarray(b)[:, 1])) == 1 or np.size(np.unique(np.asarray(b)[:, 0])) == 1:
                    intersection_points = []
                else:
                    intersection_points = intersections(a, b)
                if np.size(intersection_points) > 0 and (p, q) not in overlapping:
                    overlapping.append((p, q))
    return overlapping


def group_overlapping_ellipses(overlapping_array):
    '''
    A function that can group all of the overlapping ellipses together from an array
    of pairs.

    Parameters
    --------

    overlapping_array: The array of all the pairs of overlapping ellipses

    --------
    Returns
    --------

    An array of arrays representing all the groups of ellipses throughout the image
    '''
    groups = []
    # check to see if any of the ellipses in the pair have any other overlapping ellipses
    for pair in overlapping_array:
        in_group = 0
        in_groups = []
        for group in groups:
            for element in pair:
                if element in group:
                    in_group += 1
                    in_groups.append(groups.index(group))
        # if neither are in a group then we start a new group
        if in_group == 0:
            groups.append([pair[0], pair[1]])
        # if there is one that is in a group, then add the other ellipse to the group
        elif in_group == 1:
            for element in pair:
                if element not in groups[in_groups[0]]:
                    groups[in_groups[0]].append(element)
        # if they are both in a group, add their groups together
        elif in_group == 2:
            if in_groups[0] is not in_groups[1]:
                new_group = list(np.concatenate([groups[in_groups[0]], groups[in_groups[1]]], axis=0))
                groups.remove(groups[in_groups[0]])
                groups.remove(groups[in_groups[1]-1])
                groups.append(new_group)
    return groups


def find_ellipses(edges, n, ax, plotting=False):
    '''
    A function that can recognise ellipses from the set of grouped points in the
    image

    Parameters
    --------

    edges: The array of the image after it has been applied to the canny operator

    n: The number of ellipses that you want to find

    ax: The axis that you might want to plot the ellipses on

    plotting: A boolean value representing whether you want the ellipses to be
    plotted or not If True, the ellipses are plotted

    --------
    Returns
    --------

    An array of ellipses, with each element in the array having information about
    the centre of the ellipse, the major/minor axis lengths and the orientation.
    '''
    ellipses = []
    ellipse_arrays = []
    grouped_array = []
    for i in range(n):
        point = find_point_to_be_grouped(edges, grouped_array)
        grouped_array.append(scan_through_from_pixel(point, edges))
    # Find all the groups of points, and fit them to an ellipse
    for array in grouped_array:
        if np.shape(array)[0] > 5:
            ellipse = cv2.fitEllipse(np.array(array))
        value = 0
        # Find the properties of the ellipse
        ellipse_new = (ellipse[0][1], ellipse[0][0], ellipse[1][1], ellipse[1][0], -ellipse[2])
        for ell in ellipse_arrays:
            if np.allclose(ellipse_new, ell):
                value += 1
        if value == 0:
            ellipse_to_plot = Ellipse((ellipse[0][1], ellipse[0][0]), ellipse[1][1], ellipse[1][0], -ellipse[2])
            ellipse_arrays.append(ellipse_new)
            ellipses.append(ellipse_new)
        # if we want to plot the ellipses to show them
        if plotting == True:
            ax.add_patch(ellipse_to_plot)
    return ellipses


def get_grouped_points(edges, n, tol=20):
    '''
    Using this function, we can find all the groups of points within the image.
    If the size of the group is not large enough, then we disregard it, because
    it will cause problems when computing the overlapping points

    Parameters
    --------

    edges: The array of the image after it has been applied to the canny operator

    n: The number of groups of points that you want to find

    tol: The number of points that must be in the array for it to be considered
    valid. Default at 20 unless specified by the user

    --------
    Returns
    --------

    An array of groups of grouped points
    '''
    grouped_array = []
    for i in range(n):
        point = find_point_to_be_grouped(edges, grouped_array)
        group = scan_through_from_pixel(point, edges)
        # Providing the group is big enough we will consider it
        if np.shape(group)[0] > tol:
            grouped_array.append(group)
    return grouped_array


def get_all_phases(image):
    '''
    A function that uses the k-means algorithm to find all 3 phases in a microstructural
    image

    Parameters
    --------

    image: The array of the image for which each pixel has an intensity value. It
    is this intensity value that is separated between the pixels.

    --------
    Returns
    --------

    The array of the matrix, fibers, and voids after being separated using the
    k-means method and a scaling process.
    '''
    matrix_1 = np.zeros(np.shape(image))
    # We first differentiate the voids from the rest of the image
    clusters = k_means(image, 2)
    if proportion_of_array(clusters) > 0.5:
        for i in range(np.shape(image)[0]):
            for j in range(np.shape(image)[1]):
                if clusters[i][j] == 0:
                    matrix_1[i][j] = 1

    # Now a scaling factor is introduced to separate the matrix and the fibers
    image = image*5
    more_clusters = k_means(image, 3)
    matrix_2 = np.zeros(np.shape(image))
    if proportion_of_array(more_clusters) > 0.5:
        for i in range(np.shape(more_clusters)[0]):
            for j in range(np.shape(more_clusters)[1]):
                if more_clusters[i][j] != 1 and clusters[i][j] != 1:
                    matrix_2[i][j] = 1

    matrix_3 = np.zeros(np.shape(image))
    for i in range(np.shape(clusters)[0]):
        for j in range(np.shape(clusters)[1]):
            if matrix_2[i][j] == 0 and matrix_1[i][j] == 0:
                matrix_3[i][j] = 1

    # We now have all of the clusters we need to differentiate between them
    matrices = [matrix_1, matrix_2, matrix_3]
    fractions = []
    for matrix in matrices:
        fractions.append(proportion_of_array(matrix)/np.size(matrix))
    # The matrix will have the highest volume fraction, followed by the fibers,
    # followed by the voids
    lowest_fraction_index = fractions.index(np.min(fractions))
    voids = matrices[lowest_fraction_index]
    highest_fraction_index = fractions.index(np.max(fractions))
    matrix = matrices[highest_fraction_index]
    for number in range(np.shape(matrices)[0]):
        if number != lowest_fraction_index and number != highest_fraction_index:
            other_num = number
    fibers = matrices[other_num]
    return matrix, fibers, voids


def get_line_points(r, width, height):
    '''
    A function that can find the two points to be used for the lineal path function

    Parameters
    --------

    r: The distance that the two points will be separated by

    width: The width of the image that can be used to generate a random point

    height: The height of an image that can be used to generate a random point

    --------
    Returns
    --------

    The two points to be used in the lineal path function
    '''
    point_1 = np.array([random.uniform(0, height-r-1), random.uniform(0, width-r-1)])
    angle = random.uniform(0, 2*math.pi)
    addition = np.array([r*math.cos(angle), r*math.sin(angle)])
    point_2 = point_1 + addition
    return point_1, point_2


def is_in_group(index, grouped_ellipses):
    '''
    A function that checks if a certain ellipse is in a group of ellipses based
    on the index of the ellipse in the array of ellipses

    Parameters
    --------

    index: The index of the ellipse

    grouped ellipses: The array of the ellipses that are in a group

    --------
    Returns
    --------

    The group that the ellipse is in if the ellipse is in a group, and an empty
    array if the ellipse is not in a group
    '''
    for group in grouped_ellipses:
        # Loop through all the groups and see if they contain the index
        if index in group:
            return group
    return []


def two_point_cluster(point_1, point_2, ellipses, grouped_ellipses, plotting=False):
    '''
    A function to compute the two-point cluster function for two points separated
    by a vector |r|

    Parameters
    --------
    point_1: The first point to be checked for being in a certain phase

    point_2: The second point to be checked for being in a certain phase

    ellipses: The array of ellipses to check whether the points are in

    grouped_ellipses: The array that represents all of the ellipses that are grouped
    together

    plotting: A boolean value as to whether the results will be plotted

    --------
    Returns
    --------
    A boolean value representing whether the two points lie in the same cluster
    '''
    truth = False
    for ellipse in ellipses:
        # Find if each of the ellipses is in a group
        group = is_in_group(ellipses.index(ellipse), grouped_ellipses)
        y0 = ellipse[0]
        x0 = ellipse[1]
        a = ellipse[2]
        b = ellipse[3]
        theta = -ellipse[4]*(math.pi/180)
        # Find whether the point is in the ellipse, and if it is check the group
        if point_in_ellipse(point_1[1], point_1[0], x0, y0, a, b, theta):
            if point_in_ellipse(point_2[1], point_2[0], x0, y0, a, b, theta):
                truth = True
            elif np.size(group) != 0:
                for index in group:
                    y0 = ellipses[index][0]
                    x0 = ellipses[index][1]
                    a = ellipses[index][2]
                    b = ellipses[index][3]
                    theta = -ellipses[index][4]*(math.pi/180)
                    if point_in_ellipse(point_2[1], point_2[0], x0, y0, a, b, theta):
                        truth = True
    # If we want to plot the results, then we can either plot them red or green
    if plotting:
        if truth == True:
            plt.scatter(point_1[0], point_1[1], c='g', s=5)
            plt.scatter(point_2[0], point_2[1], c='g', s=5)
            line_between_points(point_1, point_2, 'g')
        elif truth == False:
            plt.scatter(point_1[0], point_1[1], c='r', s=5)
            plt.scatter(point_2[0], point_2[1], c='r', s=5)
            line_between_points(point_1, point_2, 'r')
    return truth


def lineal_path_func(points, ellipses, plotting=False):
    '''
    A function that computes the lineal path function for a group of points separated
    by a vector |r|

    Parameters
    --------

    points: The array of points that it is computed for, to check that they all lie
    in the same phase

    ellipses: The array of ellipses to check whether the points are in

    plotting: A boolean value as to whether the results will be plotted

    --------
    Returns
    --------
    A boolean value representing whether all of the points lie in the same phase or
    not.
    '''
    num = 0
    size = np.shape(points)[0]
    truth = False
    for point in points:
        # For each point, go through all of the ellipses.
        for ellipse in ellipses:
            y0 = ellipse[0]
            x0 = ellipse[1]
            a = ellipse[2]
            b = ellipse[3]
            theta = -ellipse[4]*(math.pi/180)
            if point_in_ellipse(point[1], point[0], x0, y0, a, b, theta):
                num += 1
                break
    # check if all of the points are in the same phase
    if num >= size:
        truth = True
    # if we want to plot the results, then plotting is True
    if plotting:
        if truth:
            plt.scatter(points[0][0], points[0][1], c='g', s=5)
            plt.scatter(points[-1][0], points[-1][1], c='g', s=5)
            line_between_points(points[0], points[-1], 'g')
        else:
            plt.scatter(points[0][0], points[0][1], c='r', s=5)
            plt.scatter(points[-1][0], points[-1][1], c='r', s=5)
            line_between_points(points[0], points[-1], 'r')
    return truth


def get_phases_from_generated_image(image, num_clusters):
    matrices = []
    clusters = k_means(image, num_clusters)
    for i in range(int(np.max(clusters))):
        array = np.zeros(np.shape(image))
        for x in range(np.shape(image)[0]):
            for y in range(np.shape(image)[1]):
                if clusters[x][y] == i:
                    array[x][y] = 1
        matrices.append(array)
    fractions = []
    for matrix in matrices:
        fractions.append(proportion_of_array(matrix)/np.size(matrix))
    # The matrix will have the highest volume fraction, followed by the fibers,
    # followed by the voids
    lowest_fraction_index = fractions.index(np.min(fractions))
    voids = matrices[lowest_fraction_index]
    highest_fraction_index = fractions.index(np.max(fractions))
    matrix = matrices[highest_fraction_index]
    for number in range(np.shape(matrices)[0]):
        if number != lowest_fraction_index and number != highest_fraction_index:
            other_num = number
    fibers = matrices[other_num]
    return matrix, fibers, voids


def two_point_correlation(image, phases):
    '''
    A function that computes the two-point correlation between two phases

    Parameters
    --------

    image: The array representing the microstructure image

    phases: strings of the two phases that you want to compute the correlation for

    Returns
    --------

    An array of the two-point correlation function for the two phases
    '''
    # Find the canny edge detection for both of the phases
    edge_1 = find_edges(image, phases[0])
    edge_2 = find_edges(image, phases[1])

    ft1 = np.fft.fft(edge_1, axis=0)
    ft2 = np.fft.fft(edge_2, axis=0)
    # The two-point correlation function is the inverse fft of the product of both ffts
    return np.fft.ifft(ft1 * ft2, axis=0).real


def get_gradient_and_intercept_between_points(point_1, point_2):
    '''
    A function that can get the gradient and the intercept between two points

    Parameters
    --------

    point_1: The x and y co-ordinates of the first point

    point_2: The x and y co-ordinates of the second point

    --------
    Returns
    --------

    The gradient and the intercept of the line separating the two points.
    '''
    # Gradient is taken horizontally, so if the points have the same y value, we
    # call the gradient 0.
    if point_2[1] - point_1[1] == 0:
        gradient = 0
    else:
        gradient = (point_2[1] - point_1[1])/(point_2[0] - point_1[0])
    intercept = point_1[1] - gradient*point_1[0]
    return gradient, intercept


def find_edges(image, phase, image_type):
    '''
    A function that implements the k-means algorithm on the image to separate the
    image into 3 phases. Then, the user specifies which phase they want to be
    returned

    Parameters
    --------
    image: The array representing the microstructure image

    phase: a string representing the phase that the user wants to be returned.
    Either 'matrix', 'fibers', or 'voids'

    --------
    Returns
    --------
    The array representing phase after being applied to the canny edge detector
    '''
    if image_type == 'generated':
        matrix, fibers, voids = get_phases_from_generated_image(image, 3)
    else:
    # Find all the clusters for each of the phases
        matrix, fibers, voids = get_all_phases(image)
    # Then find the appropriate edge detection
    if phase == 'matrix':
        return feature.canny(matrix, sigma=2)
    elif phase == 'fibers':
        return feature.canny(fibers, sigma=2)
    elif phase == 'voids':
        return feature.canny(voids, sigma=2)


def two_point_stats(image, stat, r, phase, n=100, num_ellipses=600, plotting=False, image_type='real'):
    '''
    Returns a probability value for the two point statistics. Either the two-point
    cluster function or the linear path function. Choose which value of r and which
    phase you want to compute the statistics for

    Parameters
    --------
    image: The array representing the microstructure image

    stat: Either the two_point_cluster or lineal_path_func

    r: The distance between the two points for the statistics |r|

    phase: The phase that you want the statistics to be calculated for (either matrix,
    fibers or voids)

    n: The number of iterations performed to find the probability

    num_ellipses: The number of ellipses to be found out of the grouped points

    plotting: A boolean value as to whether the results will be plotted

    --------
    Returns
    --------
    The probability representing the two-point function for a specific value of r
    '''
    # Initialise a figure in case we want to plot the results
    plt.figure()
    ax = plt.gca()
    height, width = image.shape
    # find the canny edge detection for the particular phase that you are finding
    # for
    edges = find_edges(image, phase, image_type)
    # Find the ellipses from the grouped points
    ellipses = find_ellipses(edges, num_ellipses, ax, plotting=plotting)
    tot = 0
    for i in range(n):
        # either do the two-point cluster
        if stat == two_point_cluster:
            point1, point2 = get_line_points(r, height, width)
            grouped_ellipses = group_overlapping_ellipses(find_overlapping_ellipses(ellipses))
            value = two_point_cluster(point1, point2, ellipses, grouped_ellipses, plotting=plotting)
        # or do the lineal path function
        elif stat == lineal_path_func:
            points = get_all_line_points(r, height, width)
            value = lineal_path_func(points, ellipses, plotting=plotting)
        tot += value
    if plotting:
        plt.imshow(image, cmap=plt.cm.gray)
    # return the probability of the statistical function.
    return tot/n


def varying_r(image, stat, phase, num, min_r, max_r, plotting=False):
    probabilities = []
    rs = np.linspace(min_r, max_r, num+1)
    for i in range(num):
        probability = two_point_stats(image, stat, rs[i], phase, plotting=plotting)
        probabilities.append(probability)
    plt.scatter(rs, probabilities)
    plt.xlabel('r')
    plt.ylabel('probability')
    plt.title(str(stat))
    plt.show()
    return


'''
Below is an example of how to use the code to calculate the two-point statistics
'''
# load in the image
# image = io.imread("Images/Manual_Microstructures_Gray_3/image0.png", as_gray=True)
# calculate the statistic
# probability = two_point_stats(image, two_point_cluster, 100, 'fibers', plotting=True)
# plt.show()



# do the same here for the lineal path function
# probability = two_point_stats(image, lineal_path_func, 100, 'fibers', plotting=True, num_ellipses=300, n=50, image_type='generated')
# plt.show()
