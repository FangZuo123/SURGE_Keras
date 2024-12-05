import argparse
import numpy as np
from pathlib import Path
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from model import get_model, PSNR, L0Loss, L1Loss, UpdateAnnealingParameter
from generator_US import SRImageGenerator_PA

import cv2
class Schedule:
    def __init__(self, nb_epochs, initial_lr):
        self.epochs = nb_epochs
        self.initial_lr = initial_lr

    def __call__(self, epoch_idx):
        if epoch_idx < self.epochs * 0.25:
            return self.initial_lr
        elif epoch_idx < self.epochs * 0.50:
            return self.initial_lr * 0.5
        elif epoch_idx < self.epochs * 0.75:
            return self.initial_lr * 0.25
        return self.initial_lr * 0.125


def get_args():
    parser = argparse.ArgumentParser(description="train noise2noise model",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image_dir", type=str, required=True,
                        help="train image dir")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="test image dir")
    parser.add_argument("--image_size", type=int, default=64,
                        help="training patch size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("--nb_epochs", type=int, default=60,
                        help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--steps", type=int, default=1000,
                        help="steps per epoch")
    parser.add_argument("--loss", type=str, default="mse",
                        help="loss; mse', 'mae', or 'l0' is expected")
    parser.add_argument("--weight", type=str, default=None,
                        help="weight file for restart")
    parser.add_argument("--output_path", type=str, default="checkpoints",
                        help="checkpoint dir")
    parser.add_argument("--model", type=str, default="srresnet",
                        help="model architecture ('srresnet' or 'unet')")
    parser.add_argument("--circ", type=str, default=False,
                        help="circshift or not during training")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    image_dir = args.image_dir
    test_dir = args.test_dir
    image_size = args.image_size
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    steps = args.steps
    loss_type = args.loss
    circ = args.circ
    output_path = Path(__file__).resolve().parent.joinpath(args.output_path)
    model = get_model(args.model)

    if args.weight is not None:
        model.load_weights(args.weight)

    #opt = Adam(lr=lr)
    opt = Adam(learning_rate=lr)
    callbacks = []

    if loss_type == "l0":
        l0 = L0Loss()
        callbacks.append(UpdateAnnealingParameter(l0.gamma, nb_epochs, verbose=1))
        loss_type = l0()
    if loss_type == "l1":
        l1 = L1Loss()
        callbacks.append(UpdateAnnealingParameter(l1.gamma, nb_epochs, verbose=1))
        loss_type = l1()
    model.compile(optimizer=opt, loss=loss_type, metrics=[PSNR])

    generator = SRImageGenerator_PA(image_dir=image_dir,batch_size=batch_size,image_size=image_size,image_length = 64000,circ=circ)




    val_generator = SRImageGenerator_PA(image_dir=test_dir,batch_size=batch_size,image_size=image_size,image_length=64)
    output_path.mkdir(parents=True, exist_ok=True)
    callbacks.append(LearningRateScheduler(schedule=Schedule(nb_epochs, lr)))
    callbacks.append(ModelCheckpoint(str(output_path) +  "/weights.{epoch:03d}-{val_loss:.3f}.keras",
                                     monitor="val_PSNR",
                                     verbose=1,
                                     mode="max",
                                     save_best_only=False))

    hist = model.fit(generator,
    
                               steps_per_epoch= 1000,
                               epochs=10,
                               validation_data=val_generator,
                               verbose=1,
                               callbacks=callbacks)

    np.savez(str(output_path.joinpath("history.npz")), history=hist.history)


if __name__ == '__main__':
    main()

