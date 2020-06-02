import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T
import json
import os
import glob
from dataset import BrainDataset
import torch
import utils
import pprint
from engine import train_one_epoch, evaluate
import utils

pp = pprint.PrettyPrinter(indent = 4)

annotations = json.load(open(os.path.join('./brain-tumor/data_cleaned/annotations_all.json')))
annotations = list(annotations.values())
new_annot = {a['filename']:a for a in annotations}

train_list = glob.glob('./brain-tumor/data_cleaned/train/*.jpg')
val_list = glob.glob('./brain-tumor/data_cleaned/val/*.jpg')
test_list = glob.glob('./brain-tumor/data_cleaned/test/*.jpg')

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # define training and validation data loaders
    train_loader = torch.utils.data.DataLoader(
        BrainDataset(train_list, annotations, get_transform(train = True)), batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    val_loader = torch.utils.data.DataLoader(
        BrainDataset(val_list, annotations, get_transform(train = False)), batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, val_loader, device=device)
        if epoch % 2 != 0:
            torch.save(model.state_dict(), 'checkpoints/CP{}.pth'.format(epoch))
            print("Checkpoint {} saved !".format(epoch))

    print("That's it!")

if __name__ == "__main__":
    main()