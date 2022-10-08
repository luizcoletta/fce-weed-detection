# ------------------------------------------------------------------------------
# USEFUL FUNCTIONS
# > These functions are required in different places in the code
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pylab
import random
import cv2 # OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library (https://opencv.org/)
from PIL import Image
from keras_segmentation.metrics import get_iou
import os
import imgaug as ia
import imgaug.augmenters as iaa
import os.path

def create_dir(filePath):
    if not os.path.isdir(filePath):
        os.mkdir(filePath)


def color_list(num_elements):
    random.seed(0)
    clist = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_elements)]

    '''lst_color = [(255, 255, 255),  # BACKGROUND - BGR BRANCO
                 (0, 0, 255),      # CANA - BGR VERMELHO
                 (0, 255, 255),    # ESTILHAÇO - BGR AMARELO
                 (255, 0, 255),    # RAIZ - BGR MAGENTA
                 (0, 255, 0),      # TOCO - BGR VERDE
                 (255, 0, 0)]      # TOLETE - BGR AZUL'''

    lst_color = [(255, 255, 255),  # BACKGROUND - BGR BRANCO
                 (0, 255, 0),      # GRAMINEA - BGR VERDE
                 (0, 0, 255),      # TOUCEIRA - BGR VERMELHO
                 (255, 0, 255),    #  - BGR MAGENTA
                 (0, 255, 0),      #  - BGR VERDE
                 (255, 0, 0)]      #  - BGR AZUL

    for i in range(num_elements):
        clist[i] = lst_color[i]

    return clist


def matrix2augimage(matrix, size_tuple):
    mat = matrix
    mat[mat == 0] = 255
    mat[mat < 255] = 0
    mat = cv2.merge((mat, mat, mat))
    mat = mat.astype(np.uint8)
    img_res = Image.fromarray(mat)
    img_res = img_res.resize(size_tuple, Image.ANTIALIAS)
    return img_res


def save_file(path, name_file, extension, data, num_format):
    if extension == 'txt':
        np.savetxt(path + name_file + "." + extension, data, delimiter='', fmt=num_format) # '%.4f'
    else:
        if extension == 'csv':
            np.savetxt(path + name_file + "." + extension, data, delimiter=',', fmt=num_format)
        else:
            if extension == 'png':
                save_image(path + name_file + "." + extension, data)


def save_image(name_file, data):
    show_graph = False
    if show_graph:
        width = data.shape[1]
        height = data.shape[0]
        plt.figure(figsize=(width/1000, height/1000), dpi=100)
        imgplot = plt.imshow(data)
        imgplot.set_cmap('RdYlGn')
        #min1 = NDVIc[np.isfinite(map)].min()
        #max1 = NDVIc[np.isfinite(map)].max()
        #plt.clim(min1, max1)
        plt.colorbar()
        #plt.axis('off')
        plt.title('NDVIc')
        pylab.show()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(name_file, dpi=1000)
    else:
        plt.imsave(name_file, data, dpi=1000)
        #plt.imsave(name_file, data, dpi=1000, cmap='RdYlGn')


def get_colored_segmentation_image(seg_arr, n_classes):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    colors = color_list(n_classes)

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img

'''def open_image(name_file, id):
    id_img = id
    downloaded = driveG.CreateFile({'id':id_img})
    downloaded.GetContentFile(name_file)
    img = cv2.imread(name_file)
    return img

def resize_image(img, percentual):
    scale_percent = percentual # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def NDVIc(img, mask, str_pref_file, show_img):

    name_txt_file = str_pref_file + "-ndvic.txt"
    name_csv_file = str_pref_file + "-ndvic.csv"
    name_png_file = str_pref_file + "-ndvic.png"

    ### ESTÁ CERTO R=2, G=1 E B=0???
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    NDVIc = 0.7669 - (0.0132*R) + (0.0017*G) + (0.0056*B)

    NDVIc[mask == False] = -2

    xy = np.where(NDVIc > -2)
    xy_list = list(zip(xy[0], xy[1]))
    values = []
    for xy in xy_list:
        values.append(NDVIc[xy[0], xy[1]])
    m = np.mean(values)
    v = np.var(values)

    save_file(name_txt_file, values, 'txt')
    save_file(name_csv_file, NDVIc, 'csv')
    save_file(name_png_file, NDVIc, 'png')

    img_ndvic = open_image(name_png_file, '1--lnnK3oFtSxPtLSs32bHN0_KuLYwjD-')

    blank_back = False
    if blank_back:
        R = img_ndvic[:, :, 2]
        G = img_ndvic[:, :, 1]
        B = img_ndvic[:, :, 0]
        R[mask == False] = 255
        G[mask == False] = 255
        B[mask == False] = 255
        img_ndvic = np.dstack((R, G))
        img_ndvic = np.dstack((img_ndvic, B))

    if show_img:
        cv2_imshow(resize_image(img_ndvic, 20))

    return [values, m, v]

def ExG(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    ExG = (2*g) - r - b
    np.savetxt('exg.csv', Exg, delimiter=',', fmt='%.2f')
    !cp exg.csv "/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/data"

def ExGR(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    ExGR = ExG - ((1.4*r) - g)
    np.savetxt('exgr.csv', ExGR, delimiter=',', fmt='%.2f')
    !cp exgr.csv "/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/data"

def VEG(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    VEG = g / ((r**0.667) * (b ** 0.333))
    np.savetxt('veg.csv', VEG, delimiter=',', fmt='%.2f')
    !cp veg.csv "/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/data"

def CIVE(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    CIVE = (0.441 * r) - (0.881 * g) + (0.385 * b) + 18.78745
    np.savetxt('cive.csv', CIVE, delimiter=',', fmt='%.2f')
    !cp cive.csv "/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/data"

def COM(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    COM = (0.25 * ExG) + (0.30 * ExGR) + (0.33 * CIVE) + (0.12 * VEG)
    np.savetxt('com.csv', COM, delimiter=',', fmt='%.2f')
    !cp com.csv "/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/data"

def RGBVI(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    RGBVI = ((G * G) - (R * B))/((G * G) + (R * B))
    np.savetxt('rgbvi.csv', RGBVI, delimiter=',', fmt='%.2f')
    !cp rgbvi.csv "/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/data"

def GLI(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    GLI = ((2*G)-R-B)/((2*G)+R+B)
    np.savetxt('gli.csv', GLI, delimiter=',', fmt='%.2f')
    !cp gli.csv "/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/data"

def VARI(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    VARI = (G - R) / (G + R - B)
    np.savetxt('vari.csv', VARI, delimiter=',', fmt='%.2f')
    !cp vari.csv "/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/data"

def MPRI(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    MPRI = (G - R) / (G + R)
    np.savetxt('mpri.csv', MPRI, delimiter=',', fmt='%.2f')
    !cp mpri.csv "/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/data"

def TGI(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    TGI = G - (0.39 * R) - (0.61 * B)
    np.savetxt('tgi.csv', TGI, delimiter=',', fmt='%.2f')
    !cp tgi.csv "/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/data"

def RGVBI(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    RGVBI = (G - (B * R)) / ((G * G) + (B * R))
    np.savetxt('rgvbi.csv', RGVBI, delimiter=',', fmt='%.2f')
    !cp rgvbi.csv "/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/data"

def MGVRI(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    MGVRI = ((G * G) - (R * R))/ ((G * G) + (R * R))
    np.savetxt('mgvri.csv', MGVRI, delimiter=',', fmt='%.2f')
    !cp mgvri.csv "/content/drive/MyDrive/MyFiles/PROJECTS/2020-Self-Training/Codes/data"'''


def roi_extraction(rgb, gt, labels):

    height = rgb.shape[0]
    width = rgb.shape[1]

    # Creating image with only interest regions
    img_roi = np.zeros([height, width, 3], dtype=np.uint8)
    img_roi.fill(255) # or img[:] = 255

    # Mask file: True = regions of interest
    mask = np.full((height, width), False, dtype=bool)

    for l in labels:

        ##print(">> Obtaining region " + str(l))

        result = np.where(gt == l)
        listOfCoordinates = list(zip(result[0], result[1]))

        for cord in listOfCoordinates:

            b = rgb[cord[0], cord[1], 0]
            g = rgb[cord[0], cord[1], 1]
            r = rgb[cord[0], cord[1], 2]

            img_roi[cord[0], cord[1], 0] = r
            img_roi[cord[0], cord[1], 1] = g
            img_roi[cord[0], cord[1], 2] = b

            mask[cord[0], cord[1]] = True

    return [img_roi, mask]


def iou_metric(gt, pred, num_classes):
    # https://fairyonice.github.io/Learn-about-Fully-Convolutional-Networks-for-semantic-segmentation.html
    # Using Intersection over Union (IoU) measure for each class
    # Average IoU is equal to TP/(FN + TP + FP)
    iou = get_iou(gt, pred, num_classes)
    m_iou = np.mean(iou)
    v_iou = np.var(iou)
    d_iou = np.std(iou)
    return [iou.tolist(), [m_iou, v_iou, d_iou]]

### https://github.com/aleju/imgaug
'''def data_augmentation(img_path, seg_path):

    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
    ])

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    seg = cv2.cvtColor(cv2.imread(seg_path), cv2.COLOR_BGR2RGB)[:,:,0]

    aug_det = seq.to_deterministic()
    image_aug = aug_det.augment_image(img)

    segmap = ia.SegmentationMapOnImage(seg, nb_classes=np.max(seg) + 1, shape=img.shape)
    segmap_aug = aug_det.augment_segmentation_maps(segmap)
    segmap_aug = segmap_aug.get_arr_int()

    return image_aug, segmap_aug'''


def closing_points_image(img, kernel, iter):
    # https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iter)
    return close


def find_contours_image(img):
    # cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    return cnts


def bgr2gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def threshold_image(img, threshold, max_value, type):
    thresh = cv2.threshold(img, threshold, max_value, type)
    return thresh


def smooth_image(img, ksize):
    # https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
    blur = cv2.medianBlur(img, ksize)
    return blur


def sharpen_image(img, kernel):
    sharpen = cv2.filter2D(img, -1, kernel)
    return sharpen


def resize_image(img, size):
    if not isinstance(size, tuple):
        height = int(img.shape[0] * size / 100)
        width = int(img.shape[1] * size / 100)
    else:
        height = size[0]
        width = size[1]
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img


def find_squares(img, cnts, min_area, max_area, save_squares):
    count = 0
    areas_lst = []
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(c)
            ROI = img[y:y+h, x:x+w]
            if save_squares:
                cv2.imwrite('ROI_{}.png'.format(count), ROI)
            cv2.rectangle(img, (x, y), (x + w, y + h), (36, 255, 12), 2)
            areas_lst.append(area)
            count += 1
    return areas_lst, img


def squares_in_image(img):

    min_size_sqrs = 2000 # 2000
    max_size_sqrs = 8000000 # 7000

    '''if isinstance(img, str):
        img = resize_image(cv2.imread(img), (720, 960))
    else:
        img = resize_image(img, (720, 960))'''

    # img = resize_image(cv2.imread('data/preprocessing/2.jpg'), (806, 1075))
    # img = resize_image(cv2.imread('data/preprocessing/3.jpg'), (806, 1075))

    smooth_img = smooth_image(bgr2gray(img), 5)
    sharpen_img = sharpen_image(smooth_img, np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]]))  # sharpen_kernel
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.medianBlur(gray, 5)
    # sharpen_kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
    # sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    # https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html
    # thresh = cv2.threshold(sharpen, 120, 255, cv2.THRESH_BINARY)[1]
    thresh_img = threshold_image(sharpen_img, 120, 255, cv2.THRESH_BINARY)[1]

    close_img = closing_points_image(thresh_img, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 10)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=10)

    cnts = find_contours_image(close_img)
    # cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cv2.findContours(close, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    areas_lst, img = find_squares(img, cnts, min_size_sqrs, max_size_sqrs, True)

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return areas_lst, img 

'''im_col = cv2.imread("results/typification/" + result_desc + "_colored_" + "f")
im_col [np.where(im_col == 211)] = 255
img_hsv = cv2.cvtColor(im_col, cv2.COLOR_RGB2HSV)
lab = cv2.cvtColor(im_col, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
limg = cv2.merge((cl, a, b))
final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)'''