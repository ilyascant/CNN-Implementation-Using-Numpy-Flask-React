import numpy as np

class Convolution2D:
    """
    2D convolution katman implemantasyonu.
    """
    def convolution(self, image, kernel, bias, stride=1):
        """
        Kernel'lerin (Filtrelerin) uygulanarak ozellik ortaya cikmasini saglar
        kernel kadar output cikisi olur  
        """
        (numberOfFilter, filterChannel, fx, fy) = kernel.shape
        (imageChannel, ix, iy) = image.shape

        output_shape = int((ix - fx) / stride) + 1
        output = np.zeros((numberOfFilter, output_shape, output_shape))

        for n in range(numberOfFilter):
            out_x = 0
            for x in range(0, ix - fx + 1, stride):
                out_y = 0
                for y in range(0, iy - fy + 1, stride):
                    output[n, out_x, out_y] = np.sum(np.multiply(kernel[n], image[:, x:x+fx, y:y+fy])) + bias[n, 0]
                    out_y += 1
                out_x += 1

        return output
        
    def convolutionBackward(self, dConv_prev, conv_in, kernel, stride):
        (numberOfFilter, filterChannel, fx, fy) = kernel.shape
        (imageChannel, ix, iy) = conv_in.shape

        dOut = np.zeros(conv_in.shape)
        dFilt = np.zeros(kernel.shape)
        dBias = np.zeros((numberOfFilter, 1))

        for n in range(numberOfFilter):
            out_x = 0
            for x in range(0, ix - fx + 1, stride):
                out_y = 0
                for y in range(0, iy - fy + 1, stride):
                    dOut[:, x:x+fx, y:y+fy] += dConv_prev[n, out_x, out_y] * kernel[n]
                    dFilt[n] += dConv_prev[n, out_x, out_y] * conv_in[:, x:x+fx, y:y+fy]
                    out_y += 1
                out_x += 1
        
            dBias[n] = np.sum(dConv_prev[n])

        return dOut, dFilt, dBias
