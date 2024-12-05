import argparse
import numpy as np
from pathlib import Path
import cv2
from keras.models import Model
from model import get_model

import time
from sklearn.decomposition import PCA
from keras.layers import Conv2D
import scipy.io
from scipy.io import savemat
def get_args():
    parser = argparse.ArgumentParser(description="Test trained model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="test mat dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--weight_file", type=str, required=True,
                        help="trained weight file")
    parser.add_argument("--test_noise_model", type=str, default="gaussian,25,25",
                        help="noise model for test images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="if set, save resulting images otherwise show result using imshow")
    parser.add_argument("--enlargement_factor", type=int, default=2,
                        help="Increase factor specified as N, where the channel count is scaled to 2^N times its original value.")
                                                  
    args = parser.parse_args()
    return args


def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)



def main_PA():
    args = get_args()
    image_dir = args.image_dir
    weight_file = args.weight_file
    enlargement_factor = args.enlargement_factor
    model = get_model(args.model)
    model.load_weights(weight_file)


    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths_jpg = list(Path(image_dir).glob("*.jpg"))
    image_paths_png = list(Path(image_dir).glob("*.png"))

    image_paths = image_paths_jpg + image_paths_png
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        h, w= image.shape
        image = image.reshape(h,w,1)

        print(image_path)
        h, w, _ = image.shape
        
 

        i_image = image
        for i in range(enlargement_factor):
            h, w, _ = i_image.shape

            input_image = np.zeros((h, w * 2, 1))
            input_image[:, 0:(2 * w):2, :] = i_image
            input_image[:, 1:(2 * w):2, :] = i_image

            noise_image = input_image

            pred = model.predict(np.expand_dims(noise_image, 0))

            denoised_image = pred[0]

            out_image = denoised_image
            i_image = out_image


        if args.output_dir:
            cv2.imwrite(str(output_dir.joinpath(image_path.name))[:-4] + ".png", out_image)

        else:
            cv2.imshow("result", out_image)

            if key == 113:
                return 0

if __name__ == '__main__':
    main_PA_error()

