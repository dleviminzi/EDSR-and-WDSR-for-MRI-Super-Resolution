import os
import matplotlib.pyplot as plt
from dataset import dataset, random_crop, random_flip, random_rotate
from edsr import edsr
from wdsr import wdsr_b
from train import EdsrTrainer,WdsrTrainer

depth = 16
scale = 4

weights_dir = './weights/edsr-16-x4/'
weights_file = os.path.join(weights_dir, 'weightsM.h5')
weights_dir2 = './weights/wdsr-b-32-x4/'
weights_file2=os.path.join(Weights_dir,'weights.h5')

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(WEights_dir2,exist_ok=True)

train_ds = dataset(batch_size=16, random_transform=True, subset='train')
valid_ds = dataset(batch_size=1, random_transform=False, subset='valid')

model = edsr(scale=scale, num_res_blocks=depth)
model2 = wdsr_b(scale=scale,num_res_blocks=depth)

model.load_weights('weights/edsr-16-x4/weights.h5')
model2.load_weights('weights/wdsr-b-32-x4/weights.h5')

trainer = EdsrTrainer(model=model, checkpoint_dir=f'.ckpt/edsr-{depth}-x{scale}')
trainer2 = WdsrTrainer(model=model,checkpoint_dir=f'.ckpt/wdsr-b-{depth}-x{scale}')

trainer.train(train_ds, valid_ds.take(10), steps=300000, evaluate_every=10, save_best_only=True)
trainer2.train(train_ds,valid_ds.take(10),steps=300000,evaluate_every=10,save_best_only=True)
trainer.restore()
trainer2.restore()

psnrv = trainer.evaluate(valid_ds)
print(f'PSNR(edsr) = {psnrv.numpy():3f}')

psnrw = trainer2.evaluate(valid_ds)
print(f'PSNR(wdsr)={psnr.numpy():3f}')

trainer.model.save_weights(weights_file)
trainer2.model.save_weights(weights_file2)
