import torch
import sys

sys.path.append('..')
from models.models import MMP
import warnings

warnings.filterwarnings('ignore')
from datasets.dataloader import load_data
from utils.utils import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda_id', type=int, default=0,
                    help='dataset')
parser.add_argument('-d', '--dataset', type=str, default='texas',
                    help='dataset')
args = parser.parse_args()


def basic_train(epoch, model, g, features, labels, train_mask, val_mask, test_mask,
                lr=1e-2, dur=10, weight_decay=5e-4, reg_lambda=1, loss_fun=torch.nn.CrossEntropyLoss(),
                ):
    set_seed(202)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss_list, reg_loss_list, val_loss_list, train_acc_list, val_acc_list, test_acc_list = [], [], [], [], [], []
    best_val_acc, best_test_acc = 0, 0
    dur = epoch / dur
    best_epoch = 0
    for iter in range(epoch):
        model.train()
        output = model(g, features)
        if hasattr(model, 'reg_loss'):
            reg_loss = model.reg_loss * reg_lambda
            reg_loss_list.append(reg_loss.item())
        else:
            reg_loss = 0
            reg_loss_list.append(0)
        loss = masked_loss(output, labels, train_mask, loss_fun)
        loss += reg_loss
        train_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            output = model(g, features)
            val_loss = masked_loss(output, labels, val_mask, loss_fun)
            val_loss_list.append(val_loss.item())
            train_acc, val_acc, test_acc = evaluate(output, labels, train_mask, val_mask, test_mask)
            train_acc_list.append(train_acc)
            val_acc_list.append(val_acc)
            test_acc_list.append(test_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_epoch = iter

        if (iter + 1) % dur == 0:
            train_l, reg_l, val_l = np.mean(train_loss_list), np.mean(reg_loss_list), np.mean(val_loss_list)
            print(
                "Epoch {:4d}, Train_loss {:.4f}, Reg_loss{:.4f}, Val_loss {:.4f}, train_acc {:.4f},  val_acc {:.4f}, test_acc{:.4f}".format(
                    iter + 1, train_l, reg_l, val_l, train_acc, val_acc, test_acc))
    print("Best at {} epoch, Val Accuracy {:.4f} Test Accuracy {:.4f}".format(best_epoch, best_val_acc, best_test_acc))
    return model


if __name__ == "__main__":
    device = 'cuda:{}'.format(args.cuda_id) if args.cuda_id >= 0 else 'cpu'
    dataset = args.dataset
    # one split_id for example
    # for heterophily datasets, follow the H2GCNs, do not add self_loop

    if dataset == 'texas':
        data = load_data('texas', device, split_id=3,)
        model = MMP(data.norm_noself_adj, data.x.shape[1], 64, data.num_of_class, num_layers=2, dropout=0.5,
                    reg_type='cos').to(device)
        basic_train(1000, model, data.norm_noself_adj, features=data.x, labels=data.y,
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                    lr=5e-2, weight_decay=5e-4, dur=10, reg_lambda=0.1)

    elif dataset == 'wisconsin':
        data = load_data('wisconsin', device, split_id=1,)
        model = MMP(data.norm_noself_adj, data.x.shape[1], 64, data.num_of_class, num_layers=2, dropout=0.5,
                    reg_type='cos').to(device)
        basic_train(1000, model, data.norm_noself_adj, features=data.x, labels=data.y,
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                    lr=5e-2, weight_decay=5e-4, dur=10, reg_lambda=1)

    elif dataset == 'actor':
        data = load_data('film', device, split_id=2,)
        model = MMP(data.norm_noself_adj, data.x.shape[1], 64, data.num_of_class, num_layers=2, dropout=0.5,
                    reg_type='cos').to(device)
        basic_train(1000, model, data.norm_noself_adj, features=data.x, labels=data.y,
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                    lr=5e-2, weight_decay=5e-4, dur=10, reg_lambda=0.1)

    elif dataset == 'squirrel':
        data = load_data('squirrel', device, split_id=0, )
        model = MMP(data.norm_noself_adj, data.x.shape[1], 64, data.num_of_class, num_layers=2, dropout=0.5, ).to(
            device)
        basic_train(1000, model, data.norm_noself_adj, features=data.x, labels=data.y,
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                    lr=5e-2, weight_decay=5e-5, dur=10, reg_lambda=0)

    elif dataset == 'chameleon':
        data = load_data('chameleon', device, split_id=0, )
        model = MMP(data.norm_noself_adj, data.x.shape[1], 64, data.num_of_class, num_layers=2, dropout=0.5, ).to(
            device)
        basic_train(1000, model, data.norm_noself_adj, features=data.x, labels=data.y,
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                    lr=5e-2, weight_decay=5e-4, dur=10, reg_lambda=0)

    elif dataset == 'cornell':
        data = load_data('cornell', device, split_id=2, )
        model = MMP(data.norm_noself_adj, data.x.shape[1], 64, data.num_of_class, num_layers=2, dropout=0.5,
                    reg_type='cos').to(device)
        basic_train(1000, model, data.norm_noself_adj, features=data.x, labels=data.y,
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                    lr=5e-2, weight_decay=5e-4, dur=10, reg_lambda=1)

    elif dataset == 'citeseer':
        data = load_data('citeseer', device, split_id=3, )
        model = MMP(data.norm_adj, data.x.shape[1], 64, data.num_of_class, num_layers=2, dropout=0.5, ).to(device)
        basic_train(1000, model, data.norm_adj, features=data.x, labels=data.y,
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                    lr=5e-2, weight_decay=5e-4, dur=10, reg_lambda=0.2)

    elif dataset == 'pubmed':
        data = load_data('pubmed', device, split_id=0,)
        model = MMP(data.norm_adj, data.x.shape[1], 64, data.num_of_class, num_layers=2, dropout=0.5, ).to(device)
        basic_train(1000, model, data.norm_adj, features=data.x, labels=data.y,
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                    lr=5e-2, weight_decay=5e-4, dur=10, reg_lambda=0.2)

    elif dataset == 'cora':
        data = load_data('cora', device, split_id=0,)
        model = MMP(data.norm_adj, data.x.shape[1], 64, data.num_of_class, num_layers=2, dropout=0.5).to(device)
        basic_train(1000, model, data.norm_adj, features=data.x, labels=data.y,
                    train_mask=data.train_mask, val_mask=data.val_mask, test_mask=data.test_mask,
                    lr=5e-2, weight_decay=5e-4, dur=10, reg_lambda=0)
    else:
        raise "not implement"
