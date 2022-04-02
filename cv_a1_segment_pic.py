'''
The Assignment is to write a program that accepts a scanned image file of a printed form, 
and produces an output file that contains the student's marked answers. 
There are 31 possible correct answers per question, because some questions might instruct the student
to fill in multiple options in the same question (e.g. choices A and B might both be true so the student
should mark both). 
The program should create an output file that indicates the answers that it has recognized from the student's answer sheet. 
The output file should have one line per question, with the question number, followed by a space, followed by the letter(s) that were selected by the student. 
It should also output an x on a line for which it believes student has written in an answer to the left of the question

For the task, we first split the answer sheet to smaller images, each image contains one question.
Therfore, it will be easier for recognizing the answers, and would not mixed up with other questions.

The script performs the process of splitting images.
'''
import os.path
import time
from PIL import Image, ImageEnhance
import numpy as np


def segment_preprocess(img_name, pic_route=''):
    assert img_name.endswith('.png') or img_name.endswith('.jpg')
    im = Image.open(pic_route + img_name)
    # Turn image to gray-scale
    im = im.convert("L")
    im = ImageEnhance.Color(im).enhance(10)

    im_width = im.width
    im_height = im.height
    im_array = np.array(im.getdata())
    im_matrix = [[None for i in range(im_width)] for j in range(im_height)]
    for k in range(len(im_array)):
        x_loc = k % im_width
        y_loc = k // im_width
        im_matrix[y_loc][x_loc] = im_array[k]
    return im_matrix, im_width, im_height, im


def horizontal_detect(im_height, im_matrix, quesution_interval_max):
    y_scan_start = im_height // 4
    white_y = []
    for y in range(y_scan_start, im_height):
        color_avg = np.mean(im_matrix[y])
        if color_avg > (255 * 0.995):
            white_y.append(y)
    i = 1
    white_block = []
    tmp = []
    tolerance = 1
    while i < len(white_y):
        if not tmp:
            tmp.append(white_y[i])
        elif white_y[i] <= white_y[i - 1] + 1 + tolerance:
            tmp.append(white_y[i])
        else:
            white_block.append(tmp)
            tmp = []
        i += 1
    # last line
    white_block.append([y for y in white_y if y > white_block[-1][-1]])
    white_center = [int(np.mean(b)) for b in white_block]
    sup = []
    for i in range(2, len(white_center) - 1):
        if white_center[i] - white_center[i - 1] > quesution_interval_max:
            new_center = (white_center[i] + white_center[i - 1]) // 2
            if new_center in white_y:
                sup.append(new_center)
            new_center_n = new_center
            while (new_center not in white_y) and (new_center_n not in white_y):
                new_center += 1
                new_center_n -= 1
                if new_center in white_y:
                    sup.append(new_center)
                    break
                elif new_center_n in white_y:
                    sup.append(new_center_n)
                    break
    white_center += sup
    white_center.sort()
    return white_center


def vertical_detect(im_matrix, im_width, im_height, v_tolerance, handwritten_space):
    white_vertical_line = []
    for h in range(im_width):
        if np.mean([im_matrix[y][h] for y in range(im_height // 2, im_height)]) >= 254:
            white_vertical_line.append(h)
    for j in range(len(white_vertical_line) - 1):
        if white_vertical_line[j] < white_vertical_line[j + 1] - v_tolerance - 1:
            break
    v_lines = [white_vertical_line[j]]
    j = len(white_vertical_line) - 1
    while j > 0:
        if white_vertical_line[j] > white_vertical_line[j - 1] + v_tolerance + 1:
            break
        j -= 1
    v_lines.append(white_vertical_line[j])
    v_lines = [v_lines[0], None, None, v_lines[1]]
    for i in [1, 2]:
        tmp_v = v_lines[0] + i * ((v_lines[3] - v_lines[0]) // 3)
        while True:
            if np.mean([im_matrix[y][tmp_v] for y in range(im_height // 2, im_height)]) < 252:
                break
            tmp_v -= 1
        v_lines[i] = tmp_v
    v_lines[0] = v_lines[0] - 2 * handwritten_space
    v_lines = [int(v) for v in v_lines]
    return v_lines


def subimage_main(img_name='blank_form.jpg', pic_route='CV/a1/jmpresne-etachen-andhuan-a1-main/test-images/',
                  quesution_interval_max=60, verticle_window_thre=30, v_tolerance=2, handwritten_space=30):
    im_matrix, im_width, im_height, im = segment_preprocess(img_name, pic_route)

    #######################################
    # horizontal lines
    white_center = horizontal_detect(im_height, im_matrix, quesution_interval_max)

    #######################################
    # vertical lines
    v_lines = vertical_detect(im_matrix, im_width, im_height, v_tolerance, handwritten_space)
    #######################################
    # segment to sub-images
    sub_file = '{}_segment/'.format(img_name.replace('.jpg', ''))
    if not os.path.isdir(pic_route + sub_file ):
        os.mkdir(pic_route + sub_file)
    k = 1
    n_file = 0
    for line_i in range(1, len(white_center)):
            new_img = []
            for y_ in range(white_center[line_i - 1], white_center[line_i] + 1):
                new_img += im_matrix[y_]
            if white_center[line_i] - white_center[line_i - 1] + 1 < verticle_window_thre:
                continue
            new_img = np.reshape(new_img, (white_center[line_i] - white_center[line_i - 1] + 1, im_width))
            flag = 0
            # reduce y-axis white part
            if k == 1:
                row_white = 0
                while row_white < len(new_img):
                    if np.mean(new_img[row_white]) < 253:
                        break
                    row_white += 1
                new_img = new_img[row_white:]
            elif k == 29:
                row_white = -1
                while row_white < len(new_img):
                    if np.mean(new_img[row_white]) < 253:
                        break
                    row_white -= 1
                new_img = new_img[:row_white]
            for j in range(1, 4):
                q_number = k + 29 * (j - 1)
                new_img_q = np.array([new_img[y][v_lines[j - 1]:v_lines[j]] for y in range(len(new_img))])

                if np.mean(new_img_q) > 245:
                    continue
                data = Image.fromarray(np.uint8(new_img_q), mode='L')
                data.save('{}q_{}.png'.format(pic_route + sub_file, q_number))
                n_file += 1
                flag = 1
            if flag:
                k += 1

    print('Save {} sub-images at {}'.format(n_file, pic_route + sub_file))


##########################################################
if __name__ == '__main__':
    print('Type tmg file name (endswith .jpg or .png:')
    img_name = input()
    print('Type save route: (Default: \'test-images/\')')
    pic_route = input()
    # t1 = time.time()
    subimage_main(img_name=img_name, pic_route=pic_route)
    # print('total time:', time.time()-t1)
    # for f in os.listdir(pic_route):
    #     if f.endswith('jpg'):
    #         main(img_name=f)
