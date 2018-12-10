import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img_mat

import cv2

def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]

    max_a = np.max(imga)
    max_b = np.max(imgb)

    imga = np.divide(imga, max_a)
    imgb = np.divide(imgb, max_b)
    new_imgb = np.zeros_like(imga)
    new_imgb[:,:,0] = imgb
    new_imgb[:,:,1] = imgb
    new_imgb[:,:,2] = imgb

    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=new_imgb
    return new_img


if __name__ == "__main__":

    path = '/Users/surabhibhargava/acads/cs4731/project/code/datasets/sketchy'

    photo_path = os.path.join(path, 'photo')
    sketch_path = os.path.join(path, 'sketch')

    classes = [item for item in os.listdir(photo_path) if os.path.isdir(os.path.join(photo_path, item))]

    for cls in classes:
        if "DS" in cls:
            continue

        allfiles = [f for f in os.listdir(os.path.join(photo_path,cls)) if os.path.isfile(os.path.join(photo_path, cls, f))]

        for file in allfiles:
            if "DS" in file:
                continue
            file_name = file.split(".")[0]
            ph = os.path.join(photo_path, cls,  file)
            # range depends on number of sketches for a given image. if 5 sketches per image, range(1,6)
            for i in range(1,2):
                sk = os.path.join(sketch_path, cls, file_name + "-" + str(i) + ".png" )
                # if os.path.isfile(ph) and os.path.isfile(sk):
                # imgs = map(Image.open, [ph, sk])
                imga = plt.imread(ph)
                imgb = cv2.imread(sk, cv2.CV_8UC1)

                ima = cv2.distanceTransform(imgb, cv2.DIST_L2, 3)

                concat_imgs = concat_images(imga, ima)
                save_path = os.path.join('/Users/surabhibhargava/acads/cs4731/project/code/sketchy_concat', cls)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                img_mat.imsave(save_path + "/" + file_name + "-" + str(i) + ".jpg", concat_imgs)





