import os
import cv2
import numpy as np

from tqdm import tqdm
from PIL import Image
from makeup import Makeup
from parser import get_args



def color_makeup(A_txt, B_txt, alpha):
    color_txt = model.makeup(A_txt, B_txt)
    color = model.render_texture(color_txt)
    color = model.blend_imgs(model.face, color * 255, alpha=alpha)

    return color


def pattern_makeup(A_txt, B_txt, render_texture=False):
    mask = model.get_mask(B_txt)
    mask = (mask > 0.0001).astype("uint8")
    pattern_txt = A_txt * (1 - mask)[:, :, np.newaxis] + B_txt * mask[:, :, np.newaxis]
    pattern = model.render_texture(pattern_txt)
    pattern = model.blend_imgs(model.face, pattern, alpha=1)

    return pattern


def generate_images(imgA_path, imgB_path):
    imgA = np.array(Image.open(imgA_path))
    imgB = np.array(Image.open(imgB_path))
    imgA = cv2.resize(imgA, (256, 256))
    imgB = cv2.resize(imgB, (256, 256))

    model.prn_process(imgA)
    A_txt = model.get_texture()
    B_txt = model.prn_process_target(imgB)

    output = color_makeup(A_txt, B_txt, args.alpha)
   
    x2, y2, x1, y1 = model.location_to_crop()
    output = np.concatenate([imgB[x2:], model.face[x2:], output[x2:]], axis=1)

    return output


def make_data(save_path, imgsA_path_list, imgsB_path_list):
    len_A = len(imgsA_path_list)
    len_B = len(imgsB_path_list)
    indexA = np.arange(0, len_A)
    indexB = np.arange(0, len_B)
    
    assert len(imgsA_path_list) == len(imgsB_path_list), "{}-{}".format(len(imgsA_path_list), len(imgsB_path_list))

    np.random.shuffle(indexA)  
    np.random.shuffle(indexB)  
    imgsA_path_list = imgsA_path_list[indexA]
    imgsB_path_list = imgsB_path_list[indexB]
    for i in tqdm(range(len(imgsA_path_list))):
        output = generate_images(imgsA_path_list[i], imgsB_path_list[i])
        save_img_path = os.path.join(save_path, f"{i}.png")
        Image.fromarray((output).astype("uint8")).save(save_img_path)


if __name__ == "__main__":
    seed = 789
    np.random.seed(seed) 
    imgsA_path = "/content/data/images/non-makeup"
    imgsB_path = "/content/data/images/makeup"
    save_train_path = "/content/drive/MyDrive/De-makup/data_dmk/train"
    save_val_path = "/content/drive/MyDrive/De-makup/data_dmk/val"
    save_test_path = "/content/drive/MyDrive/De-makup/data_dmk/test"

    # B > A, B is style
    imgsA_fn_list = sorted(os.listdir(imgsA_path))
    imgsB_fn_list = sorted(os.listdir(imgsB_path))
    imgsA_path_list = np.array([os.path.join(imgsA_path, fn) for fn in imgsA_fn_list])
    imgsB_path_list = np.array([os.path.join(imgsB_path, fn) for fn in imgsB_fn_list])

    np.random.shuffle(imgsA_path_list)  
    np.random.shuffle(imgsB_path_list)  

    args = get_args()
    model = Makeup(args)
    
    # Train
    make_data(save_train_path, imgsA_path_list[689:1000], imgsB_path_list[689:1000])
    # Val
    make_data(save_val_path, imgsA_path_list[1000:1100], imgsB_path_list[1000:1100])
    # Test
    make_data(save_test_path, imgsA_path_list[1000:1100], imgsB_path_list[1100:1200])
   
