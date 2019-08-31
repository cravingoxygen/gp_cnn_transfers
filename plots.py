import matplotlib.pyplot as plt

blue = (57/255.0, 106/255.0, 177/255.0)
orange = (218/255.0, 124/255.0, 48/255.0)
green = (62/255.0, 150/255.0, 81/255.0)
red = (204/255.0, 37/255.0, 41/255.0)
grey = (83/255.0, 81/255.0, 84/255.0)
purple = (107/255.0, 76/255.0, 154/255.0)
maroon = (146/255.0, 36/255.0, 40/255.0)
yellow = (148/255.0, 139/255.0, 61/255.0)
colours = [blue, orange, green, red, grey, purple, maroon, yellow]

eps = [0.01, 0.05, 0.06, 0.065, 0.07, 0.075, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
gp_success = [0.825242718446606, 17.71844660194175, 59.80582524271845, 87.71844660194175, 97.23300970873787, 99.41747572815534, 99.90291262135922, 100.0, 99.51456310679612, 99.51456310679612, 99.51456310679612, 99.51456310679612, 99.51456310679612, 99.51456310679612, 99.51456310679612, 99.51456310679612]
transfer_to_cnn = [0.5825242718446644, 0.7281553398058249, 0.825242718446606, 0.922330097087376, 1.116504854368927, 1.1650485436893177, 1.2135922330097082, 1.4077669902912593, 1.553398058252431, 3.1067961165048508, 6.165048543689322, 11.067961165048546, 19.51456310679611, 36.6504854368932, 50.873786407766985, 56.359223300970875]
cnn_success = []
transfer_to_gp = []

plt.plot(eps, gp_success, 'o-', color=blue,linewidth=2, markersize=8)
plt.plot(eps, cnn_success, 'x-', color=red,linewidth=2, markersize=8)
plt.plot(eps, transfer_to_cnn, 'x--', color=purple, linewidth=2, markersize=8)
plt.plot(eps, transfer_to_gp, 'o--', color=maroon, linewidth=2, markersize=8)
plt.show()