from references.engine import train_one_epoch, evaluate
import references.utils as utils

import os
import sys
from tqdm.notebook import tqdm

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from config import *

     
def get_detection_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.nms_thresh = 0.3
    
    return model

def evaluate_and_write_result_files(model, data_loader, output_dir):
    print(f'EVAL {data_loader.dataset}')
    model.eval()
    results = {}
    for imgs, targets in tqdm(data_loader):
        imgs = [img.to(device) for img in imgs]

    with torch.no_grad():
        preds = model(imgs)

    for pred, target in zip(preds, targets):
        results[target['image_id'].item()] = {'boxes': pred['boxes'].cpu(),
                                                'scores': pred['scores'].cpu()}
        
    data_loader.dataset.write_results_files(results, output_dir)
    data_loader.dataset.print_eval(results)

if __name__ == "__main__":
    from dataset import train_loader, train_dataset

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the model using our helper function
    model = get_detection_model(train_dataset.num_classes)
    
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.00001,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10,
                                                gamma=0.1)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=200)
        
        # update the learning rate
        lr_scheduler.step()
        
        # evaluate on the test dataset
        if epoch % 2 == 0:
            # evaluate_and_write_result_files(model, data_loader_test)
            torch.save(model.state_dict(), os.path.join("model", f"model_epoch_{epoch}.model"))