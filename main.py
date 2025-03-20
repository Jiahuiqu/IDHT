import torch, argparse, os
import HSI_model, config
from torch.utils.data import DataLoader
import loader
from test import plot_roc_curve


def distillation_loss(y_pred_student, y_soft_labels, temperature=1.0):
    loss = torch.nn.KLDivLoss()(torch.log_softmax(y_pred_student / temperature, dim=1),
                                torch.softmax(y_soft_labels / temperature, dim=1))
    return loss


def train(args):
    if torch.cuda.is_available() and config.use_gpu:
        DEVICE = torch.device("cuda:" + str(config.gpu_name2))  # config.gpu_name
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device("cpu")
    print("current deveice:", DEVICE)

    train_dataset = loader.Dataset(cemlabel='fusion_label.mat', data=args.data,
                                   mode="train_4",
                                   channel=64,
                                   padding=0)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1,
                                             drop_last=True)

    model = HSI_model.IDHT(num_class=args.num_class).to(DEVICE)
    old = torch.load('./result/Sa/model_stage3.pth', map_location=DEVICE)
    old_model = HSI_model.task3(num_class=args.num_class - 1).to(DEVICE)
    old_model.load_state_dict(old)
    state_dict = model.state_dict()
    model_dict = {}
    for k, v in old.items():
        if k in state_dict:
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict)
    loss_criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4, weight_decay=1e-6)

    for epoch in range(1, args.maxepoch):
        model.train()
        total_loss = 0
        TT_loss = 0
        kd_loss_ = 0
        for batch, (data, target) in enumerate(train_data):
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad()
            with torch.no_grad():
                old_model_feature, _ = old_model(data)
            feature, out = model(data)
            loss = loss_criterion(out, target)
            kd_loss = distillation_loss(feature, old_model_feature, temperature=2.0)
            T_loss = loss + kd_loss * 0.5
            T_loss.backward()
            optimizer.step()
            total_loss += loss.item()
            TT_loss += T_loss.item()
            kd_loss_ += kd_loss.item()

        print("epoch", epoch,
              "\n    distillation loss:", kd_loss_ / len(train_dataset) * args.batch_size,
              "\n    CROSS loss:", total_loss / len(train_dataset) * args.batch_size,
              "\n    total loss:", TT_loss / len(train_dataset) * args.batch_size)

    torch.save(model.state_dict(), os.path.join('./result/Sa/model_stage4.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IDHT')
    parser.add_argument('--batch_size', default=200, type=int, help='')
    parser.add_argument('--num_class', default=5, type=int, help='')
    parser.add_argument('--maxepoch', default=601, type=int, help='')
    parser.add_argument('--sizemage', default=512 * 217, type=int, help='')
    parser.add_argument('--data', default="./dataset/Salinas.mat", type=str, help='')
    args = parser.parse_args()
    train(args)
