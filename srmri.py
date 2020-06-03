import os
import matplotlib.pyplot as plt
from dataset import dataset, random_crop, random_flip, random_rotate
from model.edsr import edsr
from train import EdsrTrainer

depth = 16
scale = 4

weights_dir = './weights/edsr-16-x4/'
weights_file = os.path.join(weights_dir, 'weightsM.h5')

os.makedirs(weights_dir, exist_ok=True)

train_ds = dataset(batch_size=16, random_transform=True, subset='train')

valid_ds = dataset(batch_size=1, random_transform=False, subset='valid')

model = edsr(scale=scale, num_res_blocks=depth)

model.load_weights('weights/edsr-16-x4/weights.h5')

trainer = EdsrTrainer(model=model, checkpoint_dir=f'.ckpt/edsr-{depth}-x{scale}')

trainer.train(train_ds, valid_ds.take(10), steps=300000, evaluate_every=10, save_best_only=True)

trainer.restore()

psnrv = trainer.evaluate(valid_ds)
print(f'PSNR = {psnrv.numpy():3f}')

trainer.model.save_weights(weights_file)
