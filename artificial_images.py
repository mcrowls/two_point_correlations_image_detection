import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import random
import math
import pandas as pd
from matplotlib.patches import Ellipse


def ellipse_inequality(x, y, x0, y0, a, b, theta):
    lhs = (math.cos(theta)*(x-x0) + math.sin(theta)*(y-y0))**2/a**2
    rhs = (math.sin(theta)*(x-x0) + math.cos(theta)*(y-y0))**2/b**2
    return lhs+rhs


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
    #tests if a point[xp,yp] is within
    #boundaries defined by the ellipse
    #of center[x,y], diameter d D, and tilted at angle

    cosa=math.cos(angle)
    sina=math.sin(angle)
    dd=d/2*d/2
    DD=D/2*D/2

    a =math.pow(cosa*(xp-x)+sina*(yp-y),2)
    b =math.pow(sina*(xp-x)-cosa*(yp-y),2)
    ellipse=(a/dd)+(b/DD)

    if ellipse <= 1:
        return True
    else:
        return False

def point_in_ellipse(x, y, x0, y0, a, b, theta):
    truth_value = False
    if pointInEllipse(y, x, x0, y0, b, a, theta):
        truth_value = True
    return truth_value


# Purely a plotting function
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
    xs = np.array([point_1[1], point_2[1]])
    ys = np.array([point_1[0], point_2[0]])
    plt.plot(xs, ys, c=colour)
    return


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


# Access all of the points regularly spaced between two points, for the lineal_path_func
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
    point = np.array([random.uniform(0, width-r-1), random.uniform(0, height-r-1)])
    points.append(point)
    angle = random.uniform(0, 2*math.pi)
    increments = r/10
    for i in range(10):
        addition = np.array(polar_form(increments, angle))
        point = point + addition
        points.append(point)
    return np.asarray(points)


# This can draw all of the ellipses onto the image given the properties from regionprops
def draw_ellipses(csv):
    '''
    A function that can plot all of the ellipses found within the image

    Parameters
    --------

    csv: The csv file showing the properties of all the ellipses within the image
    for a certain phase

    --------
    Returns
    --------

    None, but all the ellipses are plotted onto a graph
    '''
    for i in range(np.shape(csv)[0]):
        x0 = csv.iloc[i]['centroidx']
        y0 = csv.iloc[i]['centroidy']
        a = csv.iloc[i]['major_axis_length']
        b = csv.iloc[i]['minor_axis_length']
        theta = -(180/math.pi)*csv.iloc[i]['orientation']
        ell = Ellipse((y0, x0), b, a, theta)
        ax.add_patch(ell)
    ax.imshow(image)


# This function just generates two points at either end of a line.
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
    points = []
    point_1 = np.array([random.uniform(0, height-r-1), random.uniform(0, width-r-1)])
    angle = random.uniform(0, 2*math.pi)
    addition = np.array([r*math.cos(angle), r*math.sin(angle)])
    point_2 = point_1 + addition
    return point_1, point_2


# If both points are in the same cluster of a certain phase, then return True. Else return False.
def two_point_cluster(point_1, point_2, csv, plotting=False):
    '''
    A function that computes the two-point cluster function for an image given
    two points in the image

    Parameters
    --------

    point_1: The x and y co-ordinates of the first point

    point_2: The x and y co-ordinates of the second point

    csv: The csv file that contains the information about the ellipses such as
    centroid location, major/minor axis lengths and orientations.

    plotting: A boolean value determining whether the user wants the result to
    be plotted or not

    --------
    Returns
    --------

    A boolean value which returns True if the two points are in the same cluster
    and False otherwise
    '''
    truth = False
    for i in range(np.shape(csv)[0]):
        y0 = csv.iloc[i]['centroidx']
        x0 = csv.iloc[i]['centroidy']
        a = csv.iloc[i]['major_axis_length']
        b = csv.iloc[i]['minor_axis_length']
        theta = -csv.iloc[i]['orientation']
        if point_in_ellipse(point_1[0], point_1[1], x0, y0, a, b, theta):
            if point_in_ellipse(point_2[0], point_2[1], x0, y0, a, b, theta):
                truth = True
    if plotting:
        if truth == True:
            plt.scatter(point_1[1], point_1[0], c='g', s=5)
            plt.scatter(point_2[1], point_2[0], c='g', s=5)
            line_between_points(point_1, point_2, 'g')
        elif truth == False:
            plt.scatter(point_1[1], point_1[0], c='r', s=5)
            plt.scatter(point_2[1], point_2[0], c='r', s=5)
            line_between_points(point_1, point_2, 'r')
    return truth


# If all points along a line are in the same phase (not the same cluster necessarily), return True
# Else, return False
def lineal_path_func(points, csv, plotting=False):
    '''
    A function that computes the two-point cluster function for an image given
    two points in the image

    Parameters
    --------

    points: An array of points to be inputted into the function

    csv: The csv file that contains the information about the ellipses such as
    centroid location, major/minor axis lengths and orientations.

    plotting: A boolean value determining whether the user wants the result to
    be plotted or not

    --------
    Returns
    --------

    A boolean value which returns True if all the line points are in the same phase
    and False otherwise
    '''
    num = 0
    size = np.shape(points)[0]
    truth = False
    for point in points:
        for i in range(np.shape(csv)[0]):
            y0 = csv.iloc[i]['centroidx']
            x0 = csv.iloc[i]['centroidy']
            a = csv.iloc[i]['major_axis_length']
            b = csv.iloc[i]['minor_axis_length']
            theta = -csv.iloc[i]['orientation']
            if point_in_ellipse(point[0], point[1], x0, y0, a, b, theta):
                num += 1
    if num == size:
        truth = True
    if plotting == True:
        if truth == True:
            plt.scatter(points[0][1], points[0][0], c='g', s=5)
            plt.scatter(points[-1][1], points[-1][0], c='g', s=5)
            line_between_points(points[0], points[-1], 'g')
        else:
            plt.scatter(points[0][1], points[0][0], c='r', s=5)
            plt.scatter(points[-1][1], points[-1][0], c='r', s=5)
            line_between_points(points[0], points[-1], 'r')
    return truth


def varying_r(image, stat, csv, num, min_r, max_r, plotting=False, iterations=100):
    '''
    A function that can compute one of the two-point statistics for a variation
    of r values between the points

    Parameters
    --------

    image: The image that the two-point statistics are being computed for

    stat: Either the lineal_path_func or the two_point_cluster

    csv: The csv of ellipse locations for either the fibers or the voids

    num: The number of r values to be computed for

    min_r: The minimum value of r that the probability is computed for

    max_r: The maximum value of r that the probability is computed for

    plotting: A boolean value representing if the user wants the results to be
    plotted on the image

    iterations: The number of times the function is run for each value of r so
    that a probability can be calculated.

    --------
    Returns
    --------

    None, but a graph of probability against r is plotted
    '''
    probabilities = []
    if plotting == True:
        plt.figure()
        ax = plt.gca()
        draw_ellipses(csv)
    height, width = image.shape
    rs = np.linspace(min_r, max_r, num+1)
    for i in range(num):
        tot = 0
        for t in range(iterations):
            if stat == lineal_path_func:
                points = get_all_line_points(rs[i], width, height)
                if stat(points, csv, plotting=plotting):
                    tot += 1
            elif stat == two_point_cluster:
                point1, point2 = get_line_points(rs[i], width, height)
                if stat(point1, point2, csv, plotting=plotting):
                    tot += 1
        probabilities.append(tot/iterations)

    plt.scatter(rs, probabilities)
    plt.xlabel('r')
    plt.ylabel('probability')
    plt.title(str(stat))
    plt.show()
    return


# image = io.imread("Images/Manual_Microstructures_Gray_3/image0.png")
# fiber_csv = pd.read_csv("fibres/fiber_locations_0.csv")
# varying_r(image, two_point_cluster, fiber_csv, 100, 0, 100)
