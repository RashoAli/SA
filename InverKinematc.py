import pyfabrik
from vectormath import Vector3
import numpy as np
import cv2

# Create a black image
img = np.zeros((500, 500, 3), np.uint8)

initial_joint_positions = [Vector3(0, 0, 0), Vector3(5, 2, 0), Vector3(10, 2, 0),Vector3(15, 0, 0)]
tolerance = 0.01

# Initialize the Fabrik class (Fabrik, Fabrik2D or Fabrik3D)
fab = pyfabrik.Fabrik3D(initial_joint_positions, tolerance)

currentPosition = Vector3(15, 0, 0)
goalPosition = Vector3(10, 5, 0)

newGoalPosition = currentPosition
dx = (goalPosition[0] - currentPosition[0]) / 100
dy = (goalPosition[1] - currentPosition[1]) / 100
dz = (goalPosition[2] - currentPosition[2]) / 100

img_array = []

for i in range(1, 101):
    img = np.zeros((500, 500, 3), np.uint8)
    x = newGoalPosition[0] + dx
    y = newGoalPosition[1] + dy
    z = newGoalPosition[2] + dz
    newGoalPosition = Vector3(x, y, z)
    fab.move_to(newGoalPosition)  # Return 249 as number of iterations executed
    print("--------------------------------------")
    print(i, "joints goal : ", newGoalPosition)
    print(i, "joints angels : ", fab.angles_deg)  # Holds [43.187653094161064, 3.622882738369357, 0.0]
    print(i, "joints position : ", fab.joints)

    # Draw a diagonal blue line with thickness of 5 px
    x0 = int(fab.joints[0][0] * 10 + 200)
    y0 = int(fab.joints[0][1] * 10 + 200)
    x1 = int(fab.joints[1][0] * 10 + 200)
    y1 = int(fab.joints[1][1] * 10 + 200)
    x2 = int(fab.joints[2][0] * 10 + 200)
    y2 = int(fab.joints[2][1] * 10 + 200)
    x3 = int(fab.joints[3][0] * 10 + 200)
    y3 = int(fab.joints[3][1] * 10 + 200)

    cv2.circle(img, (int(goalPosition[0] * 10 + 200), int(goalPosition[1] * 10 + 200)), 4, (0, 0, 255), 3)
    cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 3)
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), 3)
    if i % 2 == 0:
        img_array.append(img)
    #cv2.imshow('image', img)
    #cv2.waitKey(30)
    #cv2.destroyAllWindows()

# save images to video
size = (500, 500)
out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
