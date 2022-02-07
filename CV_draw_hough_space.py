'''
Draw Hough Space
According to Wikipedia (https://en.wikipedia.org/wiki/Hough_transform):
The Hough transform is a feature extraction technique used in image analysis, computer vision, and digital image processing.
The purpose of the technique is to find imperfect instances of objects within a certain class of shapes by a voting procedure. 
This voting procedure is carried out in a parameter space, from which object candidates are obtained as local maxima in a so-called accumulator space that is explicitly constructed by the algorithm for computing the Hough transform.
'''
import matplotlib.pyplot as plt
import re


def rotate(x):
    para_col = []
    for x_ in [1,0.5,0,-0.5,-1]:
        for y_ in [1,0.5,0,-0.5,-1]:
            if (x_ == 0) and (y_ == 0):
                continue
            new_p = [x[0]+x_, x[1]+y_]
            if (new_p[0] - x[0]) == 0:
                continue  # vertical line
            m = (new_p[1] - x[1]) / (new_p[0] - x[0])
            b = new_p[1] - m*new_p[0]
            para_col.append([m, b])
    return para_col
    

def draw_hough_space(x_line, title='Test):
    plt.figure()
    i = 1
    for point in x_line:
        para_ = rotate(point)
        plt.plot([c[0] for c in sorted(para_)], [c[1] for c in sorted(para_)], label='x{}'.format(i))
        i += 1
    plt.legend()
    plt.title(title)


def main_hough_transform():
    print('Input data points as [x1,y1]. When finish, type \'done\'. For demo, input \'demo\'')
    x_input = input()
    x_line = []
    while (x_input != 'done') and (x_input != 'demo'):
        xy_set = [float(c) for c in re.sub("[^0-9,]", "", x_input).split(',')]
        x_line.append(xy_set)
        print('Next point?')
        x_input = input()    
    plt.clf()
    plt.close()
    if x_input == 'done':
        draw_hough_space(x_line)

    elif x_input == 'demo':
        print('Start Demo:')
        print('\nhorizontal line')
        x_line = []
        for x1 in range(5):
            x_line.append([x1 * 2, 3])
        draw_hough_space(x_line, title='Horizontal Line')

        print('\nvertical line')
        x_line = []
        for x1 in range(5):
            x_line.append([3, x1 * 2])
        draw_hough_space(x_line, title='Vertical Line')

        print('\ncircle line')
        num_samples = 50
        theta = np.linspace(0, 2 * np.pi, num_samples)
        a, b = 1 * np.cos(theta), 1 * np.sin(theta)
        x_line = [[a[i], b[i]] for i in range(num_samples)]
        draw_hough_space(x_line, title='Circle')
    
    
###################################################################
if __name__ == '__main__':
  main_hough_transform()
