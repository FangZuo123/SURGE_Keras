from pathlib import Path
import random
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence



class SRImageGenerator_PA(Sequence):
    def __init__(self, image_dir,  batch_size=32, image_size=64,image_length= 64,circ=False):
        image_suffixes = (".jpeg", ".jpg", ".png", ".bmp")
        self.image_paths = [p for p in Path(image_dir).glob("**/*") if p.suffix.lower() in image_suffixes]
        self.image_num = len(self.image_paths)
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_length = image_length
        self.circ = circ

        if self.image_num == 0:
            raise ValueError("image dir '{}' does not include any image".format(image_dir))

    def __len__(self):
        return self.image_length

    def __getitem__(self, idx):
        batch_size = self.batch_size
        image_size = self.image_size
        x = np.zeros((batch_size, image_size, image_size, 1), dtype=np.uint8)
        y = np.zeros((batch_size, image_size, image_size, 1), dtype=np.float32)
        sample_id = 0
        

        while True:
            image_path = random.choice(self.image_paths)
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            h, w= image.shape
            image = image.reshape(h,w,1)



            if self.circ:
                j = np.random.randint(w)
                image = np.roll(image, j, axis=1)


            i = np.random.randint(h - image_size + 1)
            j = np.random.randint(w - image_size + 1)


            patch2 = image[i:(i+image_size),j:(j+image_size)]

            if random.choice([True, False]):
                patch2 = np.fliplr(patch2)

            patch1 = patch2.copy()
                
                
            patch1[:, 1:image_size:2,:] = patch2[:, 0:image_size:2,:]

            patch2 = patch2.astype(np.float32)
            
            x[sample_id] = patch1#.astype(np.float32)
            y[sample_id] = patch2#.astype(np.float32)
            sample_id += 1
            if sample_id == batch_size:
                return x, y