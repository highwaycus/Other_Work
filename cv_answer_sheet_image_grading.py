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

The script performs the process of (1) splitting images (2) recognizing and determine student's answer on the sheet images
'''
import os
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
    return pic_route + sub_file
    
    
##########################################################
# Answer Recognition
# segment sub image to 6 pieces
def segment_to_answere_scale(question_img='q_1.png', sub_img_route=''):
    im = Image.open(sub_img_route + question_img)
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
    x = im_width - 1
    white_vertical = []
    while x > 10:
        if np.mean([im_matrix[i][(x-10):x] for i in range(im_height)]) > 250:
            white_vertical.append(x)
        x -= 1
    tmp = 0
    white_block = []
    for j in range(1, len(white_vertical)):
        if white_vertical[j] < (white_vertical[j - 1] - 5):
            white_block.append(white_vertical[tmp: (j - 1)])
            tmp = j
    white_vertical_center = [np.mean(v) for v in white_block]
    white_vertical_center.sort()

    y_center = im_height//2
    x_mean = []
    thres = 130
    student_ans = []
    while ((not x_mean) and (thres <= 150)) or (not student_ans):
        x_idx = im_width - 1
        x_mean = []
        while x_idx >= 20:
            if np.mean([im_matrix[j][(x_idx - 20): x_idx] for j in range(y_center - 1, y_center + 2)]) < thres:
                x_mean = [1] + x_mean
            else:
                x_mean = [0] + x_mean
            x_idx -= 1
        student_ans = []
        marked_period = []
        for id in range(len(x_mean)):
            if x_mean[id] == 1:
                marked_period.append(id)
            else:
                if len(marked_period) > 5:
                    student_ans.append(20 + np.mean(marked_period))
                marked_period = []
        if len(marked_period) > 5:
            student_ans.append(20 + np.mean(marked_period))
        thres += 10
    if len(white_vertical_center) >= 5:
        student_ans = [s for s in student_ans if s > white_vertical_center[-5] - 5]
    alphabet_ans = []
    a_list = ['A', 'B', 'C', 'D', 'E']
    alphabet_dict = {i: a_list[i] for i in range(5)}
    for s in student_ans:
        vx = -1
        flag = 1
        while vx > -len(white_vertical_center):
            if s > white_vertical_center[vx]:
                alphabet_ans.append(alphabet_dict[5 + vx])
                flag = 0
                break
            vx -= 1
        if flag:
            alphabet_ans.append('A')
    if len(alphabet_ans) > 1:
        alphabet_ans = sorted(list(set(alphabet_ans)))
    return alphabet_ans


def test_main(sub_img_route):
    student_ans_sheet = []
    for f in os.listdir(sub_img_route):
        student_ans_sheet.append((int(f[2:-4]),  segment_to_answere_scale(f, sub_img_route)))
    student_ans_sheet.sort(key=lambda x: x[0])
    # print(student_ans_sheet)
    with open(pic_route + 'a-27_groundtruth.txt') as f:
        lines = f.readlines()
    k = 0
    for i in range(85):
        ground_an = set(p for p in lines[i].split(' ')[1].replace('\n', ''))
        rec_student = set(p for p in student_ans_sheet[i][1])
        if rec_student == ground_an:
            k += 1
        else:
            print(i + 1, 'wrong:', rec_student, ground_an)
    print(k / 85)


##########################################################
if __name__ == '__main__':
    print('Type tmg file name (endswith .jpg or .png:')
    img_name = input()
    print('Type save route: (Default: \'test-images/\')')
    pic_route = input()
    sub_img_route = subimage_main(img_name=img_name, pic_route=pic_route)
    test_main(sub_img_route)
