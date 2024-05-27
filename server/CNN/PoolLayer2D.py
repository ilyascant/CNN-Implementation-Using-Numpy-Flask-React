import numpy as np

class PoolLayer2D:
    """
    Pooling Katmani goruntunu boyutunu dusurmeye yarar 
    """
    def nanargmax(self, arr):
        idx = np.nanargmax(arr)
        idxs = np.unravel_index(idx, arr.shape)
        return idxs    

    def maxpool(self, image, kernel_size=2, stride=2):
        (imageChannel, ix, iy) = image.shape

        w = int((ix - kernel_size)/stride)+1
        h = int((iy - kernel_size)/stride)+1

        output = np.zeros((imageChannel, w, h))

        for n in range(imageChannel): 
            out_x = 0
            for x in range(0, ix - kernel_size + 1, stride):
                out_y = 0
                for y in range(0, iy - kernel_size + 1, stride):
                    output[n, out_x, out_y] = np.max(image[n, x:x+kernel_size, y:y+kernel_size])
                    out_y += 1
                out_x += 1

        return output
    

    def maxpoolBackward(self, dPool, conv_in, kernel_size, stride):
        (imageChannel, ix, iy) = conv_in.shape

        output = np.zeros(conv_in.shape)

        for n in range(imageChannel): 
            out_x = 0
            for x in range(0, ix - kernel_size + 1, stride):
                out_y = 0
                for y in range(0, iy - kernel_size + 1, stride):
                    (a, b) = self.nanargmax(conv_in[n, x:x+kernel_size, y:y+kernel_size])
                    output[n, x+a, y+b] = dPool[n, out_x, out_y]
                    out_y += 1
                out_x += 1

        return output