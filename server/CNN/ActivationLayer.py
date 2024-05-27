import numpy as np

class ActivationLayer:
    """
    Aktivasyon katmanlarini iceren Static Fonksiyon 
    """
    @classmethod
    def ReLU(cls, Z):
        return np.maximum(0, Z)
    
    @classmethod
    def softmax(cls, Z):
        out = np.exp(Z)
        return out/np.sum(out)

    @classmethod
    def sigmoid(cls, Z):
        Z_clipped = np.clip(Z, -255, 255)
        return 1 / (1 + np.exp(-Z_clipped))