import os
import sys

# This sample uses an MNIST PyTorch model to create a TensorRT Inference Engine
import model
import numpy as np

import tensorrt as trt
from cuda import cudart

sys.path.insert(1, os.path.join(sys.path[0], ".."))

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (4, 1, 28, 28)
    DYNA_INPUT_SHAPE = (-1, 1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


def populate_network(network, weights):
    # Configure the network layers based on the weights provided.
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.DYNA_INPUT_SHAPE)

    def add_matmul_as_fc(net, input, outputs, w, b):
        # m = batch size
        # k = product of each sample size
        m = input.shape[0]
        k = int(np.prod(input.shape) / m)

        assert np.prod(input.shape) == m * k

        # validating the weight size
        # weight size should be the product of input sample(k) and output dim(n)
        n = int(w.size / k)
        assert w.size == n * k
        assert b.size == n

        input_reshape = net.add_shuffle(input)
        input_reshape.reshape_dims = trt.Dims2(m, k)

        filter_const = net.add_constant(trt.Dims2(n, k), w)
        mm = net.add_matrix_multiply(
            input_reshape.get_output(0),
            trt.MatrixOperation.NONE,
            filter_const.get_output(0),
            trt.MatrixOperation.TRANSPOSE,
        )

        bias_const = net.add_constant(trt.Dims2(1, n), b)
        bias_add = net.add_elementwise(mm.get_output(0), bias_const.get_output(0), trt.ElementWiseOperation.SUM)

        output_reshape = net.add_shuffle(bias_add.get_output(0))
        output_reshape.reshape_dims = trt.Dims2(m, n)
        return output_reshape

    conv1_w = weights["conv1.weight"].numpy()
    conv1_b = weights["conv1.bias"].numpy()
    conv1 = network.add_convolution(
        input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b
    )
    conv1.stride = (1, 1)

    pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)

    conv2_w = weights["conv2.weight"].numpy()
    conv2_b = weights["conv2.bias"].numpy()
    conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
    conv2.stride = (1, 1)

    pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
    pool2.stride = (2, 2)

    fc1_w = weights["fc1.weight"].numpy()
    fc1_b = weights["fc1.bias"].numpy()
    fc1 = add_matmul_as_fc(network, pool2.get_output(0), 500, fc1_w, fc1_b)

    relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)

    fc2_w = weights["fc2.weight"].numpy()
    fc2_b = weights["fc2.bias"].numpy()
    fc2 = add_matmul_as_fc(network, relu1.get_output(0), ModelData.OUTPUT_SIZE, fc2_w, fc2_b)

    fc2.get_output(0).name = ModelData.OUTPUT_NAME

    softmax = network.add_softmax(fc2.get_output(0))
    softmax.axes = 1 << 1  # run softmax along second dim(zero index)

    network.mark_output(tensor=softmax.get_output(0))


def build_engine(weights):
    # You can set the logger severity higher to suppress messages (or lower to display more messages).
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    # For more information on TRT basics, refer to the introductory samples.
    builder = trt.Builder(TRT_LOGGER)

    # EXPLICIT_BATCH : Specify that the network should be created with an explicit batch dimension.
    # Creating a network without this flag has been deprecated.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    runtime = trt.Runtime(TRT_LOGGER)

    # max workspace size for any given layer, 1GB
    config.max_workspace_size = 1 << 30
    # Populate the network using weights from the PyTorch model.
    populate_network(network, weights)

    # dynamic shape optimization
    profile = builder.create_optimization_profile()
    profile.set_shape(ModelData.INPUT_NAME,
                      (1, 1, 28, 28),
                      (4, 1, 28, 28),
                      (32, 1, 28, 28))
    config.add_optimization_profile(profile)

    # # FP16
    # config.set_flag(trt.BuilderFlag.FP16)

    # Build and return an engine.
    plan = builder.build_serialized_network(network, config)
    return runtime.deserialize_cuda_engine(plan)

def trt_inference(engine, context, raw_data):
    data = np.array(raw_data)

    _, stream = cudart.cudaStreamCreate()

    inputH0 = np.ascontiguousarray(data.reshape(-1))
    outputH0 = np.empty(context.get_binding_shape(1), dtype=trt.nptype(engine.get_binding_dtype(1)))

    # initialize input and output data
    _, inputD0 = cudart.cudaMallocAsync(inputH0.nbytes, stream)
    _, outputD0 = cudart.cudaMallocAsync(outputH0.nbytes, stream)

    # move input to device
    cudart.cudaMemcpyAsync(inputD0,
                           inputH0.ctypes.data,
                           inputH0.nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                           stream)

    # execute
    context.execute_async_v2([int(inputD0), int(outputD0)], stream)

    # move output back to host
    cudart.cudaMemcpyAsync(outputH0.ctypes.data,
                           outputD0,
                           outputH0.nbytes,
                           cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                           stream)

    cudart.cudaStreamSynchronize(stream)

    cudart.cudaStreamDestroy(stream)
    cudart.cudaFree(inputD0)
    cudart.cudaFree(outputD0)

    return outputH0

# # Predict
# def predict(self):
#     self.network.eval()
#     correct = 0
#     torch_start = time.time_ns()

#     for data, target in self.test_loader:
#         with torch.no_grad():
#             data, target = data.to(self.device), target.to(self.device)
#         output = self.network(data)
#         pred = output.data.max(1)[1]
#         correct += pred.eq(target.data).cpu().sum()

#     torch_complete = time.time_ns()

#     print(
#         "\nTest set: Accuracy: {}/{} ({:.0f}%). Time: {:.4f} ms\n".format(
#             correct, len(self.test_loader.dataset), 100.0 * correct / len(self.test_loader.dataset), (torch_complete - torch_start) / 10e6
#         )
#     )

# Loads a random test case from pytorch's DataLoader
def load_random_test_case(model, pagelocked_buffer):
    # Select an image at random to be the test case.
    img, expected_output = model.get_random_testcase()
    # Copy to the pagelocked input buffer
    np.copyto(pagelocked_buffer, img)
    return expected_output


def main():
    # Train the PyTorch model
    mnist_model = model.MnistModel()

    if os.path.exists('mnist.pt'):
        print("Found pretrained weight!")
        mnist_model.load()
    else:
        print("No pretrained weight! Train from scratch!")
        mnist_model.learn()
        mnist_model.save()

    mnist_model.network.to("cpu")
    weights = mnist_model.get_weights()

    # Do inference with TensorRT.
    engine = build_engine(weights)

    # Build an engine, allocate buffers and create a stream.
    # For more information on buffer allocation, refer to the introductory samples.
    _, stream = cudart.cudaStreamCreate()
    context = engine.create_execution_context()

    # dynamic shape configure
    print("Set input shape", (4, 1, 28, 28))

    context.set_input_shape(ModelData.INPUT_NAME, (4, 1, 28, 28))
    context.set_binding_shape(0, (4, 1, 28, 28))

    print("Set input shape completed")

    np.random.seed(12345)
    raw_data = np.random.rand(*ModelData.INPUT_SHAPE).astype(np.float32)

    trt_output = trt_inference(engine, context, raw_data)

    trt_pred = np.argmax(trt_output)
    print(trt_output, trt_pred)

    import torch
    tor_data = torch.from_numpy(raw_data)
    tor_output = mnist_model.network(tor_data)
    tor_pred = tor_output.data.max(1)[1]

    print(tor_output, tor_pred)
    print(f"Valid? {np.allclose(tor_output.detach().numpy(), trt_output, atol=1e-04)}")


if __name__ == "__main__":
    main()
