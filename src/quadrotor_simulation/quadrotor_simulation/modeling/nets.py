import torch
from torch import nn, optim
import numpy as np


def get_MLP(params):
    return MLP(params['neurons'])


class MLP(nn.Module):
    def __init__(self, neurons):
        super().__init__()
        self.input_fc = nn.Linear(17, neurons[0])
        self.hidden_fc = nn.ModuleList()
        for i in range(len(neurons) - 1):
            self.hidden_fc.append(nn.Linear(neurons[i], neurons[i + 1]))
        self.output_fc = nn.Linear(neurons[-1], 6)

    def forward(self, x):
        x = self.input_fc(x)
        x = torch.relu(x)
        for layer in self.hidden_fc:
            x = layer(x)
            x = torch.relu(x)
        x = self.output_fc(x)
        return x


def prepare_tensor(inputs):
    return torch.from_numpy(np.hstack(inputs)).float()


def normalize_tensor(tensor, mean=None, std=None):
    if (mean is None):
        mean = tensor.mean(0)
    if (std is None):
        std = tensor.std(0)

    return (tensor - mean)/std, mean, std


def denormalize_tensor(tensor, mean, std):
    return tensor*std + mean


def train(net, inputs, targets, shuffle_seed=-1,  fold=-1,  val_size=0.2, optimizer=None, criterion=None, num_epochs=1000, patience=10, restore_weights=True, verbose=True, print_every=10, batch_size=32, regularizer=None, device='cpu'):

    if optimizer is None:
        optimizer = optim.Adam(net.parameters(), lr=0.0001)

    if criterion is None:
        criterion = nn.MSELoss()

    net = net.to(device)
    all_params = torch.cat([x.view(-1) for x in net.parameters()])
    print('Number of parameters:', len(all_params))

    indices = np.arange(len(inputs))
    if shuffle_seed != -1:
        np.random.seed(shuffle_seed)
        np.random.shuffle(indices)

    train_size = int(len(inputs) * (1 - val_size))
    val_size = len(inputs) - train_size

    if fold != -1:
        valid_indices = indices[fold * val_size:(fold + 1) * val_size]
        train_indices = np.concatenate((indices[:fold * val_size], indices[(fold + 1) * val_size:]))
    else:
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]

    inputs_train = inputs[train_indices]
    targets_train = targets[train_indices]
    inputs_val = inputs[valid_indices].to(device)
    targets_val = targets[valid_indices].to(device)

    best_loss = float('inf')
    best_epoch = 0
    best_weights = None
    counter = 0

    train_loss = []
    train_crit = []
    val_crit = []

    num_batches = int(np.ceil(len(inputs_train) / batch_size))
    print(
        f'Starting training, num batches: {num_batches}, num epochs: {num_epochs}, fold: {fold}, train_size: {len(inputs_train)}, val size: {len(inputs_val)}')
    print(f'train_indices: {train_indices}, val_indices: {valid_indices}, intersection: {len(np.intersect1d(train_indices, valid_indices))}, union: {len(np.union1d(train_indices, valid_indices))}')
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        epoch_loss = 0
        epoch_crit = 0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            inputs_batch = inputs_train[start_idx:end_idx]
            targets_batch = targets_train[start_idx:end_idx]

            inputs_batch = inputs_batch.to(device)
            targets_batch = targets_batch.to(device)

            outputs = net(inputs_batch)
            loss = criterion(outputs, targets_batch)
            epoch_crit += loss.item()
            if regularizer is not None:
                if 'L1' in regularizer:
                    loss += regularizer['L1'] * torch.norm(all_params, 1)
                if 'L2' in regularizer:
                    loss += regularizer['L2'] * torch.norm(all_params, 2)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        val_loss = 0
        with torch.no_grad():
            val_outputs = net(inputs_val)
            val_loss = criterion(val_outputs, targets_val)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_weights = net.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience or epoch == num_epochs - 1:
                    print('Early stopping at epoch', epoch)
                    if (restore_weights):
                        print(f'Restoring best weights from epoch {best_epoch} and stopping training')
                        net.load_state_dict(best_weights)
                    break
        # if(epoch>0):
        train_loss.append(epoch_loss / num_batches)
        train_crit.append(epoch_crit / num_batches)
        val_crit.append(val_loss.item())
        if epoch % print_every == 0 and verbose:
            print(f'Epoch {epoch} train total loss: {epoch_loss / num_batches:.4f}, train loss: {epoch_crit/ num_batches:.4f}, val loss: {val_loss:.4f}, best val loss: {best_loss:.4f}  at epoch {best_epoch}')

    print(f'Finished training at epoch {epoch}, Best val loss: {best_loss:.4f} at epoch {best_epoch} ... fold {fold}')
    net = net.to('cpu')
    return net, train_loss, train_crit, val_crit
