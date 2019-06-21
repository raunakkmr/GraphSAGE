from math import ceil
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import node_classification
from layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator
import models
import utils

def main():
    config = utils.parse_args()

    if config['cuda'] and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    dataset_args = (config['task'], config['dataset'], config['dataset_path'],
                    'train', config['num_layers'], config['self_loop'],
                    config['normalize_adj'], config['transductive'])
    dataset = utils.get_dataset(dataset_args)
    loader = DataLoader(dataset=dataset, batch_size=config['batch_size'],
                        shuffle=True, collate_fn=dataset.collate_wrapper)
    input_dim, output_dim = dataset.get_dims()

    agg_class = utils.get_agg_class(config['agg_class'])
    model = models.GraphSAGE(input_dim, config['hidden_dims'], output_dim, 
                             agg_class, config['dropout'],
                             config['num_samples'], device)
    model.to(device)

    if not config['load']:
        criterion = utils.get_criterion(config['task'])
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                            weight_decay=config['weight_decay'])
        epochs = config['epochs']
        print_every = config['print_every']
        num_batches = int(ceil(len(dataset) / config['batch_size']))
        model.train()
        print('--------------------------------')
        print('Training.')
        for epoch in range(epochs):
            print('Epoch {} / {}'.format(epoch+1, epochs))
            running_loss = 0.0
            num_correct, num_examples = 0, 0
            for (idx, batch) in enumerate(loader):
                features, node_layers, mappings, rows, labels = batch
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                out = model(features, node_layers, mappings, rows)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    running_loss += loss.item()
                    predictions = torch.max(out, dim=1)[1]
                    num_correct += torch.sum(predictions == labels).item()
                    num_examples += len(labels)
                if (idx + 1) % print_every == 0:
                    running_loss /= print_every
                    accuracy = num_correct / num_examples
                    print('    Batch {} / {}: loss {}, accuracy {}'.format(
                        idx+1, num_batches, running_loss, accuracy))
                    running_loss = 0.0
                    num_correct, num_examples = 0, 0
        print('Finished training.')
        print('--------------------------------')

        if config['save']:
            print('--------------------------------')
            directory = os.path.join(os.path.dirname(os.getcwd()),
                                    'trained_models')
            if not os.path.exists(directory):
                os.makedirs(directory)
            fname = utils.get_fname(config)
            path = os.path.join(directory, fname)
            print('Saving model at {}'.format(path))
            torch.save(model.state_dict(), path)
            print('Finished saving model.')
            print('--------------------------------')

    if config['load']:
        directory = os.path.join(os.path.dirname(os.getcwd()),
                                 'trained_models')
        fname = utils.get_fname(config)
        path = os.path.join(directory, fname)
        model.load_state_dict(torch.load(path))
    dataset_args = (config['task'], config['dataset'], config['dataset_path'],
                    'test', config['num_layers'], config['self_loop'],
                    config['normalize_adj'], config['transductive'])
    dataset = utils.get_dataset(dataset_args)
    loader = DataLoader(dataset=dataset, batch_size=config['batch_size'],
                        shuffle=False, collate_fn=dataset.collate_wrapper)
    criterion = utils.get_criterion(config['task'])
    print_every = config['print_every']
    num_batches = int(ceil(len(dataset) / config['batch_size']))
    model.eval()
    print('--------------------------------')
    print('Testing.')
    running_loss, total_loss = 0.0, 0.0
    num_correct, num_examples = 0, 0
    total_correct, total_examples = 0, 0
    for (idx, batch) in enumerate(loader):
        features, node_layers, mappings, rows, labels = batch
        features, labels = features.to(device), labels.to(device)
        out = model(features, node_layers, mappings, rows)
        loss = criterion(out, labels)
        running_loss += loss.item()
        total_loss += loss.item()
        predictions = torch.max(out, dim=1)[1]
        num_correct += torch.sum(predictions == labels).item()
        total_correct += torch.sum(predictions == labels).item()
        num_examples += len(labels)
        total_examples += len(labels)
        if (idx + 1) % print_every == 0:
            running_loss /= print_every
            accuracy = num_correct / num_examples
            print('    Batch {} / {}: loss {}, accuracy {}'.format(
                idx+1, num_batches, running_loss, accuracy))
            running_loss = 0.0
            num_correct, num_examples = 0, 0
    total_loss /= num_batches
    total_accuracy = total_correct / total_examples
    print('Loss {}, accuracy {}'.format(total_loss, total_accuracy))
    print('Finished testing.')
    print('--------------------------------')

if __name__ == '__main__':
    main()