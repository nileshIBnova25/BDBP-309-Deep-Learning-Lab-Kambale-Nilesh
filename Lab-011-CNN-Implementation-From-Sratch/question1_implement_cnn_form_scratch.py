'''By the end of this lab, you should be able to
1. Learn the idea of convolution operation along with various kernels and sizes.
2. familiarize yourself with the maxpool operation and how it helps in reducing the
dimensionality.3. Understand how the input sizes gets transformed after convolution
and max pooling operations.'''
#--------------------Imports-------------------------------#
import numpy as np
#----------------------------------------------------------#

# Create a 32x32 grayscale image


#---------------------------------------------------------------#
def con_2d(image,kernel,stride=1, padding=0):
    # Add zero padding around the image if required
    if padding > 0:
        image = np.pad(
            image,
            pad_width=padding,
            mode='constant',
            constant_values=0
        )
    # Get image and kernel dimensions
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate output dimensions
    output_height = ((image_height - kernel_height) // stride) + 1
    output_width = ((image_width - kernel_width) // stride) + 1

    # Create an empty output feature map
    output = np.zeros((output_height,output_width))

    # Slide the kernel across the image
    for i in range(output_height):
        for j in range(output_width):

            # Extract the current image region
            region = image[
                i * stride : i * stride + kernel_height,
                j * stride : j * stride + kernel_width
            ]

            # Element-wise multiplication and summation
            output[i,j]=np.sum(region * kernel)
    return output
#---------------------------------------------------------------------------#



#---------------------------------------------------------------------------#
def max_pool_2d(feature_map, pool_size=2, stride=2):
    # Get imput dimensions
    height, width = feature_map.shape

    # Calculate output dimensions
    output_height = (
        (height - pool_size) // stride
    ) + 1

    output_width = (
        (width - pool_size) // stride
    ) + 1

    output = np.zeros((output_height,output_width))

    # Slide pooling window over the feature map
    for i in range(output_height):
        for j in range(output_width):
            # Extract current pooling region
            region = feature_map[
                i * stride : i * stride + pool_size,
                j * stride : j * stride + pool_size
            ]

            # Select maximum value
            output[i, j] = np.max(region)
    return output
#--------------------------------------------------------------------#


def main():
    image = np.random.randint(0, 256, size=(32, 32))

    # Define a 3x3 kernel for vertical edge detection
    kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    #-----------------------------------------#
    print("Input image shape:", image.shape)
    print("Kernel shape:", kernel.shape)

    feature_map = con_2d(image, kernel, stride=1, padding=0)
    print("Feature map shape:", feature_map.shape)
    print(feature_map)

    #--------------------------------------------------------#
    pooled_output = max_pool_2d(feature_map, pool_size=2, stride=2)
    print("Before max pooling:", feature_map.shape)
    print("After max pooling", pooled_output.shape)
    print(pooled_output)

if __name__ == '__main__':
    main()
