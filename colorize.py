import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import colorsys
import cv2

def yiq_to_rgb(y, i, q):
    r = y + 0.948262*i + 0.624013*q
    g = y - 0.276066*i - 0.639810*q
    b = y - 1.105450*i + 1.729860*q
    r[r < 0] = 0
    r[r > 1] = 1
    g[g < 0] = 0
    g[g > 1] = 1
    b[b < 0] = 0
    b[b > 1] = 1
    return (r, g, b)

original = cv2.imread("picture\example.bmp")
marked = cv2.imread("picture\example_marked.bmp")

original = original.astype(float)/255
marked = marked.astype(float)/255

isColored = abs(original - marked).sum(2) > 0.01

(Y,_,_) = colorsys.rgb_to_yiq(original[:,:,0],original[:,:,1],original[:,:,2])
(_,I,Q) = colorsys.rgb_to_yiq(marked[:,:,0],marked[:,:,1],marked[:,:,2])

YUV = np.zeros(original.shape)
YUV[:,:,0] = Y
YUV[:,:,1] = I
YUV[:,:,2] = Q


n = YUV.shape[0]
m = YUV.shape[1]
image_size = n*m

indices_matrix = np.arange(image_size).reshape(n,m,order='F').copy()

wd = 1
nr_of_px_in_wd = (2*wd + 1)**2
max_nr = image_size * nr_of_px_in_wd

row_inds = np.zeros(max_nr, dtype=np.int64)
col_inds = np.zeros(max_nr, dtype=np.int64)
vals = np.zeros(max_nr)

# ----------------------------- Interation ----------------------------------- #

length = 0
pixel_nr = 0


# iterate over pixels in the image
for j in range(m):
    for i in range(n):
        
        # If current pixel is not colored
        if (not isColored[i,j]):
            window_index = 0
            window_vals = np.zeros(nr_of_px_in_wd)
            
            # Then iterate over pixels in the window around [i,j]
            for ii in range(max(0, i-wd), min(i+wd+1,n)):
                for jj in range(max(0, j-wd), min(j+wd+1, m)):
                    
                    # Only if current pixel is not [i,j]
                    if (ii != i or jj != j):
                        row_inds[length] = pixel_nr
                        col_inds[length] = indices_matrix[ii,jj]
                        window_vals[window_index] = YUV[ii,jj,0]
                        length += 1
                        window_index += 1
            
            center = YUV[i,j,0].copy()
            window_vals[window_index] = center

            variance = np.mean((window_vals[0:window_index+1] - np.mean(window_vals[0:window_index+1]))**2)
            sigma = variance * 0.6
            
            # Indeed, magic
            mgv = min(( window_vals[0:window_index+1] - center )**2)            
            if (sigma < ( -mgv / np.log(0.01 ))):
                sigma = -mgv / np.log(0.01)                                     
            if (sigma < 0.000002):                                              # avoid dividing by 0
                sigma = 0.000002
            
            window_vals[0:window_index] = np.exp( -((window_vals[0:window_index] - center)**2) / sigma )
            window_vals[0:window_index] = window_vals[0:window_index] / np.sum(window_vals[0:window_index])
            vals[length-window_index:length] = -window_vals[0:window_index]
        
        # Add the values for the current pixel
        row_inds[length] = pixel_nr
        
        col_inds[length] = indices_matrix[i,j]
        vals[length] = 1
        length += 1
        pixel_nr += 1

# ------------------------ After Iteration Process --------------------------- #
vals = vals[0:length]
col_inds = col_inds[0:length]
row_inds = row_inds[0:length]

# ------------------------------- Sparseness --------------------------------- #
A = sparse.csr_matrix((vals, (row_inds, col_inds)), (pixel_nr, image_size))
b = np.zeros((A.shape[0]))

colorized = np.zeros(YUV.shape)
colorized[:,:,0] = YUV[:,:,0]

color_copy_for_nonzero = isColored.reshape(image_size,order='F').copy()
colored_inds = np.nonzero(color_copy_for_nonzero)

for t in [1,2]:
    curIm = YUV[:,:,t].reshape(image_size,order='F').copy()
    b[colored_inds] = curIm[colored_inds]
    new_vals = linalg.spsolve(A, b)
    colorized[:,:,t] = new_vals.reshape(n, m, order='F')

# ------------------------------ Back to RGB --------------------------------- #

(R, G, B) = yiq_to_rgb(colorized[:,:,0],colorized[:,:,1],colorized[:,:,2])
colorizedRGB = np.zeros(colorized.shape)
colorizedRGB[:,:,0] = R
colorizedRGB[:,:,1] = G
colorizedRGB[:,:,2] = B

cv2.namedWindow("colored")
cv2.imshow("colored",colorizedRGB)
cv2.waitKey (0)