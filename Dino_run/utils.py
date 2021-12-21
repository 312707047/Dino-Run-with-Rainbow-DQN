import numpy as np
import torch

class ImageTensorProcessor:
    def to_array(self, state):
        state = np.array(state).transpose((2, 0, 1))
        state = np.ascontiguousarray(state, dtype=np.float32) / 255
        return state
    
    def to_tensor(self, state):
        state = self.to_array(state)
        state = torch.from_numpy(state)
        return state.unsqueeze(0)