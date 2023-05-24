import cv2
import numpy as np
import csv

def read_filter_points(file_path):
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points

def read_filter(filter_name):
    filter = {}
    img = cv2.imread(filter_name + ".png", cv2.IMREAD_UNCHANGED)
    filter["img"] = img
    with open(filter_name + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        filter['points'] = points
    with open(filter_name + "_hull.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        hull = []
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[0]), int(row[1])
                hull.append((x, y))
            except ValueError:
                continue
        filter['hull'] = hull
    with open(filter_name + "_hullIndex.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        hullIndex = []
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x = int(row[0])
                hullIndex.append([x])
            except ValueError:
                continue
        filter['hullIndex'] = np.array(hullIndex)
    with open(filter_name + "_tri.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        tri = []
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y, z = int(row[0]), int(row[1]), int(row[2])
                tri.append((x,y,z))
            except ValueError:
                continue
        filter['tri'] = tri
    return filter
    

def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex