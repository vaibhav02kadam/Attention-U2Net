import glob
import os

import numpy as np
import torch
from PIL import Image
from skimage import io
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from model import *

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset


# normalize the predicted SOD probability map

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d - mi) / (ma - mi)

    return dn


def save_output(image_name, pred, d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir + imidx + '.png')


def main():
    # --------- 1. get image & model path and name ---------
    model_name = 'attu2net'  # 'u2net' 'attu2net'
    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_images_DUTS-TE')  
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    print(img_name_list)

    # --------- 2. dataloader ---------
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320),  
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if model_name == 'attu2net':
        net = AttU2Net(3, 1)
    elif model_name == 'attu2netboth':
        net = AttU2NetBoth(3, 1)
    elif model_name == 'attu2netoutside':
        net = AttU2NetOutside(3, 1)
    elif model_name == 'attu2net_normal':
        net = AttU2Net_normal(3, 1)
    elif model_name == 'attu2netboth_normal':
        net = AttU2NetBoth_Normal(3, 1)
    elif model_name == 'attu2netoutside_normal':
        net = AttU2NetOutside_Normal(3, 1)
    elif model_name == 'u2net':
        net = U2NET(3, 1)
    else:
        print("Incorrect model defined Please try again")
        net=0

    if torch.cuda.is_available():
        checkpoint = torch.load(model_dir)
        net.load_state_dict(checkpoint['net_state_dict'])
        net.cuda()
    else:
        checkpoint = torch.load(model_dir, map_location='cpu')
        net.load_state_dict(checkpoint['net_state_dict'])

    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7


if __name__ == "__main__":
    main()
