#Import packages and load up an easy sklearn dataset to play around with
import numpy as np
from numba import cuda
import math
from sklearn.datasets import load_diabetes

x, y = load_diabetes(return_X_y=True, as_frame=False)

print(f'Features shape: {x.shape}')
print(f'Range of features: {x.min()} - {x.max()}')
print(f'Labels shape: {y.shape}')
print(f'Range of labels: {y.min()} - {y.max()}')

#CPU versions of some basic pre-processing, loop back to GPU implementations once the basics are done. 
def min_max_scaler(x):
    x_scaled = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    return x_scaled

def std_scaler(x):
    x_scaled = (x - x.mean(axis=0)) / x.std(axis=0)
    return x_scaled

#Perform batching on the CPU dataset
def create_batches(x, y, batch_size=64):

    if y.ndim == 1:
        y = y.reshape((y.shape[0], 1))
    else:
        pass

    pad_length = batch_size - (x.shape[0] % batch_size)
    num_batches = (x.shape[0] + pad_length) // batch_size

    x_pad_array = np.zeros((pad_length, x.shape[1]))
    x_padded = np.concatenate((x, x_pad_array), axis=0)
    x_batched = x_padded.reshape((num_batches, batch_size, x_padded.shape[1]))

    y_pad_array = np.zeros((pad_length, y.shape[1]))
    y_padded = np.concatenate((y, y_pad_array), axis=0)
    y_batched = y_padded.reshape((num_batches, batch_size, y_padded.shape[1]))

    return x_batched, y_batched

x_scaled = std_scaler(x)
y_scaled = std_scaler(y)

batch_size = 64
x_batched, y_batched = create_batches(x_scaled, y_scaled, batch_size=batch_size)

#Defining a class to hold weights and biases for linear layers on the GPU - when instantiated, creates new objects which hold weight and bias matrices of appropriate shape on GPU memory 
class NumbaLinearLayer():
    def __init__(self, batch_size, input_dim, output_dim):

        self.gpu_weights = cuda.device_array_like(np.random.normal(0, 1, (input_dim, output_dim)).astype(np.float64))
        self.gpu_bias = cuda.device_array_like(np.random.normal(0, 1, (batch_size, output_dim)).astype(np.float64))

#Standard CUDA matmul kernel implementation 
@cuda.jit
def matmul_kernel(A, B, C):
    #Get thread indices - at a lower level, .grid implements:
    #row = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    #col = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    row, col = cuda.grid(2)

    #Check bounds of indices:
    if row < C.shape[0] and col < C.shape[1]:
        #Perform computation by incrementing the 0 vals implemented in the output location when the kernel was called:
        tmp = 0.0
        for i in range(A.shape[1]): #Walks down the cols of A, which is equal to the rows of B, keeping the proper positions on the outside for output
            tmp += A[row, i] * B[i, col]
        C[row, col] = tmp

#Instantiate linear layer on GPU - stores weights and biases of appropriate shapes on GPU
linear_layer = NumbaLinearLayer(batch_size, 10, 32)
#Allocates device space for linear layer's output - matches the num_batches and batch_size of inputs, with final dimension matching output of linear layer
layer_output = cuda.device_array_like(np.zeros((x_batched.shape[0], batch_size, linear_layer.gpu_weights.shape[1])))

#Passes through a single "epoch" of matmuls on the layer over the CPU input tensor
for i in range (x_batched.shape[0]):
    #Take batch from CPU tensor and move to device
    batch = cuda.to_device(x_batched[i, :, :])
    #Allocate memory for output from batch operations
    batch_output = cuda.device_array_like(np.zeros((batch.shape[0], 32)))

    #Sets parameters to invoke numba kernel
    threads_per_block = (16,16)
    blocks_per_grid_x = math.ceil(batch_output.shape[0] / threads_per_block[0])
    blocks_per_grid_y = math.ceil(batch_output.shape[1] / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    #Performs matrix multiplication with inputs A and B as batch and layer weights, and output C as the space allocated earlier for batch outputs
    matmul_kernel[threads_per_block, blocks_per_grid](batch, linear_layer.gpu_weights, batch_output)

    #Drops the results of the matmul, stored on device, to the corresponding batch location on the pre-allocated layer output tensor
    layer_output[i, :, :] = batch_output

layer_output_cpu = layer_output.copy_to_host()




#Now we'll try to chain some forward passes, expanding input dimensions before squeezing down to target dimensions:
linear_1 = NumbaLinearLayer(batch_size, 10, 32)
linear_2 = NumbaLinearLayer(batch_size, 32, 64)
linear_3 = NumbaLinearLayer(batch_size, 64, 32)
linear_4 = NumbaLinearLayer(batch_size, 32, 16)
output_layer = NumbaLinearLayer(batch_size, 16, 1)

#Allocate memory of proper shape for our target tensor:
pred_outputs = cuda.device_array_like(np.zeros(y_batched.shape))

#Perform the passes on the input CPU tensor
for i in range(num_batches):
    #Move batch to device
    batch = cuda.to_device(x_batched[i, :, :])

    #Allocate space for outputs of all layer passes in advance - carries same batch size consistently the whole time!
    x_1 = cuda.device_array_like(np.zeros((batch_size, linear_1.gpu_weights.shape[1])))
    x_2 = cuda.device_array_like(np.zeros((batch_size, linear_2.gpu_weights.shape[1])))
    x_3 = cuda.device_array_like(np.zeros((batch_size, linear_3.gpu_weights.shape[1])))
    x_4 = cuda.device_array_like(np.zeros((batch_size, linear_4.gpu_weights.shape[1])))
    batch_output = cuda.device_array_like(np.zeros((batch_size, output_layer.gpu_weights.shape[1])))

    #PERFORM LAYER PASSES BY INVOKING AND RUNNING KERNELS SEQUENTIALLY:
    #First layer pass - takes batch, applies linear_1 weights, results go into x_1
    threads_per_block = (8,8)
    blocks_per_grid_x = math.ceil(x_1.shape[0] / threads_per_block[0]) # 64 / 8 = 8
    blocks_per_grid_y = math.ceil(x_1.shape[1] / threads_per_block[1]) # 32 / 8 = 4
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) #(8,4)
    matmul_kernel[threads_per_block, blocks_per_grid](batch, linear_1.gpu_weights, x_1)
    #Second layer pass - takes x_1, applies linear_2 weights, results go into x_2
    threads_per_block = (8,8)
    blocks_per_grid_x = math.ceil(x_2.shape[0] / threads_per_block[0]) # 64 / 8 = 8
    blocks_per_grid_y = math.ceil(x_2.shape[1] / threads_per_block[1]) # 64 / 8 = 8
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) # (8,8)
    matmul_kernel[threads_per_block, blocks_per_grid](x_1, linear_2.gpu_weights, x_2)
    #Third layer
    threads_per_block = (8,8)
    blocks_per_grid_x = math.ceil(x_3.shape[0] / threads_per_block[0]) # 64 / 8 = 8
    blocks_per_grid_y = math.ceil(x_3.shape[1] / threads_per_block[1]) # 32 / 8 = 4
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) # (8,4)
    matmul_kernel[threads_per_block, blocks_per_grid](x_2, linear_3.gpu_weights, x_3)
    #Fourth layer
    threads_per_block = (8,8)
    blocks_per_grid_x = math.ceil(x_4.shape[0] / threads_per_block[0]) # 64 / 8 = 8
    blocks_per_grid_y = math.ceil(x_4.shape[1] / threads_per_block[1])  #16 / 8 = 2
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    matmul_kernel[threads_per_block, blocks_per_grid_x](x_3, linear_4.gpu_weights, x_4)
    #Output layer
    threads_per_block = (8, 1)
    blocks_per_grid_x = math.ceil(batch_output.shape[0] / threads_per_block[0]) # 64 / 8 = 8
    blocks_per_grid_y = math.ceil(batch_output.shape[1] / threads_per_block[1]) # 1 / 1 = 1
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    matmul_kernel[threads_per_block, blocks_per_grid](x_4, output_layer.gpu_weights, batch_output)

    #Now we assign the final batch output to our destination tensor on the GPU
    pred_outputs[i, :, :] = batch_output

#The epoch passes with all operations taking place on GPU - at the end, we consolidate the result to CPU
pred_outputs_cpu = pred_outputs.copy_to_host()
