from mpl_toolkits.mplot3d import Axes3D

img_dim = 28

x = np.arange(0,img_dim)
y = np.arange(0,img_dim)
xx, yy = np.meshgrid(x, y)
x, y = xx.ravel(), yy.ravel()
z = inf0.reshape(img_dim*img_dim)
zbot = np.zeroes_like(z)
width = depth = 1

fig = plt.figure(figsize=(8, 3))

ax2 = fig.add_subplot(122, projection='3d')
ax1 = fig.add_subplot(121, projection='3d')

ax1.bar3d(x, y, zbot, width, depth, z, shade=True)


#https://matplotlib.org/3.1.1/gallery/mplot3d/3d_bars.html?highlight=bar3d
