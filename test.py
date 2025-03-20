import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch, argparse, os
import HSI_model, config
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import loader
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from sklearn.neighbors import KNeighborsClassifier
import joblib
import scipy.io


def plot_roc_curve(y, pred, title):
    # ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    fpr, tpr, threshold = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    print('roc_auc:', roc_auc)


def test(args):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name))
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    test_dataset = loader.Dataset(cemlabel='cem_3',
                                  data=args.data,
                                  mode="test_3",
                                  channel=64,
                                  padding=0)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1,
                                            drop_last=True)
    model = HSI_model.IDHT(num_class=args.num_class).to(DEVICE)
    # model = torch.nn.DataParallel(model)
    model.load_state_dict(
        torch.load('./result/Sa/model_stage3.pth', map_location=DEVICE))

    model.eval()
    with torch.no_grad():
        total_num = 0
        for batch, (data, target) in enumerate(test_data):
            print("start eval")
            data = data.to(DEVICE)
            feature, out = model(data)
            total_num += data.size(0)
            print(total_num)
            plot_roc_curve(target / (args.num_class - 1), out[:, args.num_class - 1].cpu())
            scipy.io.savemat('./result/Sa/Sa_3.mat',
                             {'Sa_3': out.cpu().detach().numpy()})
            print('over')

            # PLRS
            pred_1 = out.argmax(dim=1).cpu().reshape(512, 217).numpy()
            mat_file = scipy.io.loadmat("./dataset/Sa_cem_4.mat")
            existing_predictions = mat_file['cem_4']
            for k in range(1, args.num_class):
                condition = (pred_1 == k)
                existing_predictions[condition] = pred_1[condition]
            scipy.io.savemat('./dataset/fusion_label.mat', {'Sa_cem_4321': existing_predictions})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--batch_size', default=512 * 217, type=int, help='')
    parser.add_argument('--num_class', default=4, type=int, help='')
    parser.add_argument('--data', default="./dataset/Salinas.mat", type=str, help='')
    args = parser.parse_args()
    test(args)
