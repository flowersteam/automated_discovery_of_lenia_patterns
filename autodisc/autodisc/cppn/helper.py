import numpy as np


def create_image_cppn_input(output_size, is_distance_to_center=True, is_bias=True, input_borders=((-1, 1), (-1, 1))):

    img_height = output_size[0]
    img_width = output_size[1]

    num_of_input = 2
    if is_distance_to_center:
        num_of_input = num_of_input + 1
    if is_bias:
        num_of_input = num_of_input + 1

    cppn_input = np.zeros((img_height * img_width, num_of_input))

    min_i = input_borders[0][0]
    max_i = input_borders[0][1]
    range_i = max_i - min_i

    min_j = input_borders[1][0]
    max_j = input_borders[1][1]
    range_j = max_j - min_j

    input_idx = 0
    for i in range(img_height):

        # i coordinate in [-1 1]
        #i_input = -1.0 + 2.0 * i / (img_height - 1)
        i_input = min_i + i * range_i / (img_height - 1)


        for j in range(img_width):

            # j coordinate in [-1 1]
            #j_input = -1.0 + 2.0 * j / (img_width - 1)
            j_input = min_j +  j * range_j / (img_width - 1)

            row = []

            if is_bias:
                row.append(1.0)

            row.append(i_input)
            row.append(j_input)



            # distance to center of image
            if is_distance_to_center:
                d = np.linalg.norm([i_input, j_input])

                row.append(d)

            cppn_input[input_idx, :] = np.array(row)

            # input[input_idx, :] = [i_input, j_input]
            input_idx = input_idx + 1

    return cppn_input


def calc_neat_forward_image_cppn_output(neat_net, cppn_input,):

    output = np.zeros(np.shape(cppn_input)[0])

    for idx, input in enumerate(cppn_input):
        output[idx] = neat_net.activate(input)[0]

    return output


def calc_neat_recurrent_image_cppn_output(neat_net, cppn_input, n_iter=1):

    output = np.zeros(np.shape(cppn_input)[0])

    for idx, input in enumerate(cppn_input):

        neat_net.reset()

        for _ in range(n_iter):
            output[idx] = neat_net.activate(input)[0]

    return output


def postprocess_image_cppn_output(image_size, cppn_output):
    '''
    Each output pixel x of the CPPN is reformated by: 1-abs(x).
    And the output is reshaped into the given image dimensions.
    '''

    image = np.reshape(cppn_output, image_size)

    return 1-np.abs(image)