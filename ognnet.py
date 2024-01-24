import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import math
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from ignnet import Block, normalize_adj_matrix


class OGNNet(nn.Module):
    def __init__(self, input_dim, num_features, adj, num_classes):
        super(OGNNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)

        self.block1 = Block(64, 64)

        self.block2 = Block(64, 128)

        self.fc4 = nn.Linear(64 + 128, 256)
        self.bn1 = nn.BatchNorm1d(num_features)

        self.block3 = Block(256, 256)

        self.bn2 = nn.BatchNorm1d(num_features)

        self.block4 = Block(256, 512)

        self.bn3 = nn.BatchNorm1d(num_features)

        self.fc7 = nn.Linear(256 + 512, 256)

        self.MLP1 = nn.Linear((64 + 256 + 256) * num_features, 1024)
        self.MLP2 = nn.Linear(1024, num_classes)

        self.adj = adj
        self.num_classes = num_classes

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.batch_adj = None


    def load_batch_adj(self, x_in):
        bs = x_in.shape[0]

        adj_3d = np.zeros((bs, self.adj.shape[0], self.adj.shape[1]), dtype=float)

        for i in range(bs):
            adj_3d[i] = self.adj.cpu()

        adj_train = torch.FloatTensor(adj_3d)
        self.batch_adj = adj_train.to(x_in.device)

    def gnn_forward(self, x_in):
        self.load_batch_adj(x_in)

        x = self.fc1(x_in)

        x1 = self.relu(torch.bmm(self.batch_adj, x))
        x1 = self.block1(x1)

        x2 = self.relu(torch.bmm(self.batch_adj, x1))
        x2 = self.block2(x2)

        x3 = self.relu(torch.bmm(self.batch_adj, x2))

        x4 = torch.cat((x3, x1), 2)
        x4 = self.fc4(x4)
        x4 = self.bn1(x4)

        x5 = self.relu(torch.bmm(self.batch_adj, x4))
        x5 = self.block3(x5)
        x5 = self.bn2(x5)

        x6 = self.relu(torch.bmm(self.batch_adj, x5))
        x6 = self.block4(x6)

        x7 = self.relu(torch.bmm(self.batch_adj, x6))
        x7 = torch.cat((x7, x4), 2)
        x7 = self.bn3(self.fc7(x7))

        x = torch.cat((x7, x4, x1), 2)

        return x

    def forward(self, x_in):
        x = self.gnn_forward(x_in)

        if self.num_classes > 2:
            x = self.MLP1(x.view(x.size(0), x.size(1)*x.size(2)))
            x = self.MLP2(x)
            x = self.softmax(x)
        else:
            x = self.MLP1(x.view(x.size(0), x.size(1)*x.size(2)))
            x = self.MLP2(x)
            x = self.sigmoid(x)

        return x

    def predict(self, x_in):
        return self.forward(x_in)

    def get_local_importance(self, x_in):
        x = self.gnn_forward(x_in)

        x = x.view(x.size(0), x.size(1))

        return x.cpu().data.numpy()


def train_model(input_dim, adj_matrix, train_dataloader, val_dataloader,
                data_name, num_classes, num_epochs=300,
                learning_rate=1e-03, normalize_adj=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    adj_matrix = torch.FloatTensor(adj_matrix)
    if normalize_adj:
        adj_matrix = normalize_adj_matrix(adj_matrix)

    if not os.path.exists(f'{data_name}'):
        os.makedirs(f'{data_name}')

    if num_classes > 2:
        loss_function = nn.MSELoss()
    else:
        loss_function = nn.BCELoss()

    gnn_model = OGNNet(input_dim, adj_matrix[0].shape[0], adj_matrix.to(device), num_classes).to(device)

    optimizer_train = torch.optim.Adam(gnn_model.parameters(), lr=learning_rate)

    best_roc = 0.5

    for epoch in range(1, num_epochs + 1):
        gnn_model.train()

        train_loss = 0
        train_count = 0

        train_labels = []

        for i, data in enumerate(tqdm(train_dataloader)):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer_train.zero_grad()
            outputs = gnn_model(inputs)

            if num_classes > 2:
                loss = loss_function(outputs, labels)
                preds = torch.max(outputs, dim=-1)[1]
                train_count += torch.sum(preds == torch.max(labels, dim=-1)[1])
            else:
                loss = loss_function(outputs.reshape(-1), labels.float())
                preds = (outputs.reshape(-1) > 0.5) * 1
                train_count += torch.sum(preds == labels.data)

            train_loss += loss.item()  # * output.shape[0]

            loss.backward()
            optimizer_train.step()
            train_labels.extend(labels.tolist())

            torch.cuda.empty_cache()

        gnn_model.eval()

        val_count = 0

        list_prediction = []
        val_labels = []
        list_prob_pred = []

        for i, data in tqdm(list(enumerate(val_dataloader))):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = gnn_model(inputs)

            if num_classes > 2:
                # val_loss = loss_function(outputs, labels)
                preds = torch.max(outputs, dim=-1)[1]
                val_count += torch.sum(preds == torch.max(labels, dim=-1)[1])
                val_labels.extend(torch.max(labels, dim=-1)[1].tolist())
                list_prob_pred.extend(outputs.tolist())
            else:
                # val_loss = loss_function(outputs.reshape(-1), labels.float())
                preds = (outputs.reshape(-1) > 0.5) * 1
                val_count += torch.sum(preds == labels.data)
                val_labels.extend(labels.tolist())

            list_prediction.extend(preds.tolist())

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        if num_classes > 2:
            roc = roc_auc_score(
                val_labels,
                list_prob_pred,
                multi_class="ovr",
                average="weighted",
            )
        else:
            roc = roc_auc_score(val_labels, list_prediction)

        prec = precision_score(val_labels, list_prediction, average='macro')
        recall = recall_score(val_labels, list_prediction, average='macro')
        f_score = f1_score(val_labels, list_prediction, average='macro')

        acc = "{:.3f}".format(val_count / len(val_labels))

        if roc >= best_roc:
            best_roc = roc
            roc = "{:.3f}".format(roc)
            torch.save(gnn_model.state_dict(), f'{data_name}/ognnet-epoch[{epoch}]-{roc}.model')
            torch.save(optimizer_train.state_dict(), f'{data_name}/ognnet-epoch[{epoch}]-{roc}.optm')

        print('Acc at dev is : {}'.format(acc))
        print('ROC is : {},  prec {},  recall {}, f-score {}'.format(roc, prec, recall, f_score))
        print('Acc at epoch : {} is : {}, loss : {}'.format(epoch,
                                                            train_count / len(train_labels), train_loss))
    roc = "{:.3f}".format(roc)
    torch.save(gnn_model.state_dict(), f'{data_name}/ognnet-epoch[{epoch}]-{roc}.model')
    torch.save(optimizer_train.state_dict(), f'{data_name}/ognnet-epoch[{epoch}]-{roc}.optm')
    return gnn_model
