import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm

# Kianoush Aqabakee
# Apache License 2.0


N = 32*32
P = 10
N_sqrt = np.sqrt(N).astype('int32')
def image_to_np(path):
    im = Image.open(path)
    im_np = np.asarray(im)
    try:
        im_np = im_np[:, :, 0]
    except IndexError:
        pass
    im_np = np.where(im_np<128, -1, 1)
    im_np = im_np.reshape(N)
    return im_np
PATH="data/"
epsilon = np.asarray([image_to_np(os.path.join(PATH, '0.jpg')),
                     image_to_np(os.path.join(PATH, '1.jpg')),
                     image_to_np(os.path.join(PATH, '2.jpg')),
                     image_to_np(os.path.join(PATH, '3.jpg')),
                     image_to_np(os.path.join(PATH, '4.jpg')),
                     image_to_np(os.path.join(PATH, '5.jpg')),
                     image_to_np(os.path.join(PATH, '6.jpg')),
                     image_to_np(os.path.join(PATH, '7.jpg')),
                     image_to_np(os.path.join(PATH, '8.jpg')),
                     image_to_np(os.path.join(PATH, '9.jpg'))])                   
epsilon[0].reshape(N_sqrt, N_sqrt)
fig = plt.figure(figsize = (8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
for num, i in enumerate(epsilon):
    plt.subplot(4, 3,num+1)
    plt.imshow(np.where(i.reshape(N_sqrt, N_sqrt)<1, 0, 1), cmap='gray')
random_pattern = np.random.randint(P)
test_array = epsilon[random_pattern]
noise = np.random.normal(0,1,test_array.size)
test_array=test_array+noise

plt.figure(2)
plt.imshow(test_array.reshape(N_sqrt, N_sqrt), cmap='gray')
h = np.zeros((N))
epsilon=epsilon.reshape([epsilon.shape[0],epsilon.shape[1],1])
w=np.ones([N,N])
for i in range(P):
    w=w*(epsilon[i]+epsilon[i].T)
w=w/2**P
w=w.T*w
 
y=w@test_array
plt.figure(3)
plt.imshow(np.where(y.reshape(N_sqrt, N_sqrt)<1, 0, 1), cmap='gray')
print(random_pattern)
plt.show()
# sc.savemat('w.mat', dict(w=w))
