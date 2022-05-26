import os.path
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import os
import os.path
from PIL import Image
from PIL.ImageFile import ImageFile
from rembg.bg import remove
import io


from utils import save_file, squares_in_image


def extract_objects_from_images(images_path, dataset_name):

    files_list = [f for f in listdir(images_path) if isfile(join(images_path, f))]

    if not os.path.isdir(images_path + '/' + dataset_name):
        os.mkdir(images_path + '/' + dataset_name)

    # Uncomment the following line if working with trucated image formats (ex. JPEG / JPG)
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    for file in files_list:
        images_file = images_path + '/' + file
        print(images_file)

        f_image = np.fromfile(images_file)
        result = remove(f_image, model_name='u2net', alpha_matting=True,   # u2net_human_seg, u2net, u2netp
                        alpha_matting_foreground_threshold=240,
                        alpha_matting_background_threshold=10,
                        alpha_matting_erode_structure_size=10,
                        alpha_matting_base_size=1000)
        img = Image.open(io.BytesIO(result)).convert("RGBA")
        img_black = Image.new("RGB", img.size, "BLACK")
        img_black.paste(img, mask=img)
        img_black.save(images_path + '/' + dataset_name + '/' + file)

        #areas_lst, img = squares_in_image(images_path + '/' + dataset_name + '/' + file)

        #if not os.path.isdir(images_path + '/' + dataset_name + '/objs'):
        #    os.mkdir(images_path + '/' + dataset_name + '/objs')

        #img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #img.save(images_path + '/' + dataset_name + '/sqr-' + file)

        #img = cv2.imread(images_path + '/' + dataset_name + '/sqr-' + file, cv2.COLOR_BGR2RGB)

        img = cv2.imread(images_path + '/' + dataset_name + '/' + file, cv2.COLOR_BGR2RGB)
        img_1 = img[:, :, 1]
        mat = img_1
        mat[mat > 0] = int(file[-5:-4])

        if not os.path.isdir(images_path + '/' + dataset_name + '/annotations'):
            os.mkdir(images_path + '/' + dataset_name + '/annotations')

        #save_file(images_path + '/' + dataset_name + '/annotations/', file, 'csv', mat, '%.1f')

        im = Image.fromarray(mat)
        im.save(images_path + '/' + dataset_name + '/annotations/' + file)