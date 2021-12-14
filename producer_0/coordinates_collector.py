import cv2
import math
PI = 3.14159265359
map = cv2.imread('map_0.jpeg', 1)
coordinates = []
xcenter = 457
ycenter = 414
width_internal = 1089
height_internal = 807

width = 1089
height = 807

def convert_xy_longlat(x,y):

    # xPoint=xcenter- (width_internal/2-x)
    # yPoint=ycenter -(height_internal/2-y)
    #
    # C = (180 / (2 * PI)) * 2
    # M = (xPoint/C) - PI
    # N =-(yPoint/C) + PI
    #
    # lon_Point =math.degrees(M)
    # lat_Point =math.degrees( (math.atan( math.e**N)-(PI/4))*2 )
    # print(x)
    # print(y)

    if x < width/2 :
        lon_Point = (x * 180) / width
    else:
        lon_Point = ((x - (width/2)) * -180) / width

    if y > height/2:
        lat_Point = ((y - height/2) * 90) / height
    else:
        lat_Point = (y * -90) / height
    #
    return lon_Point,lat_Point

def click_event(event, x, y, flags, params):

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        long, lat = convert_xy_longlat(y,x)
        coordinates.append(tuple((long,lat)))


        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(map, str(x) + ',' + str(y), (x,y), font,
                    1, (20, 0, 220), 2)
        cv2.imshow('World Map', map)

    # checking for right mouse clicks
    if event==cv2.EVENT_RBUTTONDOWN:

        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = map[y, x, 0]
        g = map[y, x, 1]
        r = map[y, x, 2]
        cv2.putText(map, str(b) + ',' + str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (20, 0, 220), 2)
        cv2.imshow('World Map', map)


def collect():
    # displaying the image
    cv2.imshow('World Map', map)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('World Map', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()

class Coordinates():
    def __init__(self, coordinates):
        self.coordinates = coordinates

coordinates_obj = Coordinates(coordinates)