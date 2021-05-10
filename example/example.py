# Train and Validate
from lcfcn import lcfcn, lcfcn_loss
from PIL import Image
import torch

model = lcfcn.LCFCN(n_classes=1, lr=1e-5, device='cpu')

batch = {}
batch['images'] = lcfcn.transform_image(Image.open('input.png'))[None]
batch['points'] = torch.zeros(batch['images'].shape[-2:])[None]
point_list = [(75, 100), (75, 180),
              (75, 230), (75, 290), (80, 350)]
for y,x in point_list:
  batch['points'][:, y, x] = 1

# train for several iterations
for i in range(1000):
    loss = model.train_on_batch(batch)
    val_dict = model.val_on_batch(batch)
    print(i, '- loss:', float(loss['train_loss']), val_dict)
    if i % 10 == 0:
        # visualize blobs and heatmap
        model.vis_on_batch(batch, savedir_image='result.png')