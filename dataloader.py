
from PIL import Image
import cv2
import open3d as o3d
from random import randint, random


from data_utils import *

def comput_normals(pcd):
    o3d.geometry.estimate_normals(
        pcd,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                          max_nn=30))
    return pcd


def do_rotation(key, point_cloud, person, person_list):
    rot = randint(0,180)
    point_cloud = point_cloud.rotate([0, rot, 0], False)

    if key in person_list:
        person = person.rotate([0, rot, 0], False)

    else:
        person = ""
    return key+"_01", point_cloud, person

def do_crop_rotation(key, point_cloud, person, person_list):
    rot = randint(0, 180)
    crop = random.random()/2.5+0.9
    # print(crop)
    point_cloud = point_cloud.rotate([0, rot, 0], False)
    point_cloud = point_cloud.scale(crop, False)

    if key in person_list:
        person = person.rotate([0, rot, 0], False)
        person = person.scale(crop, False)
    else:
        person = ""
    return key+"_02", point_cloud, person

def write_res(path, key, image, text):
    tresh = randint(0,100)
    f = open(path+'allData/' + key + '.txt', "a")
    f.write(text)
    f.close()
    im = Image.fromarray(image)
    im.save(path + 'allData/' + key + '.jpg')
    if tresh>13:
        f = open(path+'train/labels/' + key + '.txt', "a")
        f.write(text)
        f.close()
        im.save(path+'train/images/' + key + '.jpg')
    else:
        f = open(path + 'test/labels/' + key + '.txt', "a")
        f.write(text)
        f.close()
        im.save(path + 'test/images/' + key + '.jpg')

def show_image_labels(im, x, y, w, h, siz):
    print( x, y, w, h, siz)
    for i in range(siz):
        for j in range(siz):
            if i > x - w / 2 and i < x + w / 2 and j > y - h / 2 and j < y + h / 2:
                im[i, j, 2] = 100
    cv2.imshow('image', im)
    cv2.waitKey(0)

def rescaleImg(image, flip = False, vertflip=False):
    # Rescale to 0-255 and convert to uint8
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                               dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    if flip:
        norm_image = np.fliplr(norm_image)
    if vertflip:
        norm_image = np.flipud(norm_image)
    return norm_image


if __name__ == '__main__':

    fallen_persons = load_list_point_cloud('./segmented_fallen_people/', 14, 18, True)
    scenes_pcd  = load_list_point_cloud('./raw/kinect2_pcd/', 12, 16)

    bounding_boxes = get_bounding_boxes(fallen_persons)
    file_list = scenes_pcd.keys()
    images_scene_data = {}


    box_data = ""
    box_data_rot = ""


    # --------------get augmentation---------------------------------
    siz = 640
    flip = True
    vertflip = True

    for file in file_list:
        # try:
        person = ""
        #compute normals
        # o3d.geometry.estimate_normals(
        #         scenes_pcd[file],
        #         search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
        #                                                           max_nn=30))
        #check the availability of the persons bounding box
        if file in bounding_boxes.keys():
            person = fallen_persons[file]

        #do crop and rotation, you can use here only rotation if you want
        key_rotation, scene_rotation, person_rotation = do_rotation(file, scenes_pcd[file], person , bounding_boxes.keys())

        #compute 5-channels birds eye view image from 6d point cloud(3d points and 3d normals
        #xm; ym is a centre of image, further we will move image from the center to the corner using this values
        xm, ym = 0, 0
        img_scene_rot, xm, ym = point_cloud_2_birdseye(scene_rotation.points, features_=scenes_pcd[file].normals, isArray=False)

        #do flipping and rescaling of the image( Rescale to 0-255 and convert to uint8)
        im = rescaleImg(img_scene_rot, flip, vertflip)
        im_copy = im.copy()
        print('fdsfs')
        #if the bounding box is exist we make the same transformation as for scene and save it's coordinate and size in the form of  string
        if file in bounding_boxes.keys():
            bou_box = get_bounding_box(person_rotation)
            img_bound_box_rot, xm, ym = point_cloud_2_birdseye(bou_box, isArray=True, xm=xm, ym=ym)
            x, y, w, h = bounding_box_by_image(img_bound_box_rot)
            if flip:
                y = siz - y
            if vertflip:
                x = siz - x
            box_data_rot = "0 " + str(y / siz) + " " + str(x / siz) + " " + str(h / siz) + " " + str(w / siz)

            # SHOW the image and corresponding bounding box

            show_image_labels(im[:1], x, y, w, h, siz)
            show_image_labels(im[:2], x, y, w, h, siz)
            show_image_labels(im[-3:], x, y, w, h, siz)
            exit()

            # imp = rescaleImg(person)
            # im_v = cv2.vconcat([im, imp])
            # cv2.imshow('image', im)
            # cv2.waitKey(0)

        else:
            box_data_rot = ""

        #the first two coordinates are the density and height of the image(after flipping, rotation, birdseye view atc.).
        # And the last 3 are normal vectors to each point.
        # Here we get the normals from this variable and save the result further as a picture to a file

        norm_im  = im_copy[:,:,-3:]
        write_res("data/normals/augment/", key_rotation, norm_im, box_data_rot)

        # except Exception:
        #     print("something is bad in the file:", file)
        #     pass











