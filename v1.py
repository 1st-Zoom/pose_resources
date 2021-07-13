import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from skimage.transform import resize

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
img = plt.imread('person_1206523.jpg')
temp = resize(img, (257, 257, 3))


input_data = np.expand_dims(temp, axis=0)
input_data = input_data.astype('float32')


# scale down factor for the image to be used as an input to the network
'''scale_factor = 0.5
new_size = int(257*scale_factor)
input_data = tf.image.resize(input_data, [new_size, new_size])'''

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_stride = 32 # this is used inside the model to decide the various height and width of the layers used in the neural network. 

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
heatmaps = interpreter.get_tensor(output_details[0]['index'])
offsets = interpreter.get_tensor(output_details[1]['index'])
# print(output_data[0][0][:5][:])
scores = tf.math.sigmoid(heatmaps)

def argmax_2d(tensor):

  # input format: BxHxWxD
  # assert rank(tensor) == 4

  # flatten the Tensor along the height and width axes
  flat_tensor = tf.reshape(tensor, (tf.shape(tensor)[0], -1, tf.shape(tensor)[3]))

  # argmax of the flat tensor
  argmax = tf.cast(tf.argmax(flat_tensor, axis=1), tf.int32)

  # convert indexes into 2D coordinates
  argmax_x = argmax // tf.shape(tensor)[2]
  argmax_y = argmax % tf.shape(tensor)[2]

  # stack and return 2D coordinates
  return tf.stack((argmax_x, argmax_y), axis=1)

heatmapPositions = argmax_2d(scores)
heatmapPositions = tf.reshape(heatmapPositions, [2,17])
heatmapPositions = tf.transpose(heatmapPositions)

offsetVectors = []
for i in range(17):
	offsetVector = [offsets[0][heatmapPositions[i][0]][heatmapPositions[i][1]][i], offsets[0][heatmapPositions[i][0]][heatmapPositions[i][1]][i+17]]
	offsetVectors.append(offsetVector)

offsetVectors = np.array(offsetVectors)

keypointPositions = heatmapPositions*output_stride + offsetVectors

confidence_scores = []
for i in range(17):
	confidence_scores.append(tf.make_ndarray(tf.make_tensor_proto(scores[0][heatmapPositions[i][0]][heatmapPositions[i][1]][i])))

print(heatmaps.shape)
print(offsets.shape)
print(scores.shape)
print(heatmapPositions.shape)
print(offsetVectors.shape)
print(keypointPositions)
print(confidence_scores)
# print(input_shape)
# print(input_details)
# print(output_details)


