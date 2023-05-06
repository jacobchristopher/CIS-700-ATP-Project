"""
Installation Packages:
  - PyTorch
  - MatPlotLib
  - thop: https://pypi.org/project/thop/
"""

import torch as tr
import matplotlib.pyplot as plt
import time
import dataset as ds
import model as mdl
import torchmetrics
from thop import profile
from dataset import encode
import random
import os
import sys

device = tr.device("cuda")

def train(model, epochs=60, data_size=2500, lr=0.01, loss_fn=None, optimizer=None, write_log=False, log_filename="log.txt"):

    # Set default loss and optimizer
    if loss_fn == None: loss_fn = tr.nn.CrossEntropyLoss()
    if optimizer == None: optimizer = tr.optim.SGD(model.parameters(), lr=lr)

    accuracy = torchmetrics.Accuracy('binary').to(device)


    print("Generating dataset...")
    train_set, val_set, test_set, embedder = ds.dataset_builder(data_size)

    # Setup metrics
    train_loss_cumulative = []        # Track the loss to graph
    val_loss_cumulative = []
    train_acc_cumulative = []
    val_acc_cumulative = []
    grad_norm = []

    start_time = time.time()    # Track CPU time

    print("Begining training loop...")
    # Run the training loop
    for epoch in range(epochs):

        run_loss = 0.0
        iter_acc = []
        iter_grad = []
        
        idx = 0
        # Iterate over batches of data
        for conjecture, step, labels in train_set:
            
            # conjecture = encode(conjecture, 256, embedder)
            # step = encode(step, 256, embedder)
            conjecture = embedder(conjecture)
            step = embedder(step)

            # Zero the gradients
            optimizer.zero_grad()
        
            if epoch == 0 and idx == 0:
                flops, params = profile(model, inputs=(conjecture, step), verbose=False)
                print("==============================================================")
                print(f"  FLOPs Per Batch: {int(flops)}, Model Parameters: {int(params)}")
                print("==============================================================")

            # Forward pass
            labels = labels.to(device)
            outputs = model(conjecture, step).to(device)
            loss = loss_fn(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            grads = tr.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
            iter_grad.append(grads.norm(p=1).cpu())
            
            # Update run loss
            run_loss += loss.item() * step.size(0)

            # Update Accuracy
            label_simpl = tr.where(labels[0] > labels[1], tr.tensor([1]).to(device), tr.tensor([0]).to(device))
            output_result = tr.where(outputs[0] > outputs[1], tr.tensor([1]).to(device), tr.tensor([0]).to(device))
            train_acc = accuracy(label_simpl, output_result)
            iter_acc.append(train_acc)
            idx += 1

        # Run Test Loss
        val_loss, val_acc = test(model, loss_fn, val_set, embedder)
        epoch_loss = run_loss / len(train_set)
        train_acc = sum(iter_acc) / len(iter_acc)        

        # Update cumulative loss
        train_loss_cumulative.append(epoch_loss)
        val_loss_cumulative.append(val_loss)

        train_acc_cumulative.append(train_acc.cpu())
        val_acc_cumulative.append(val_acc.cpu())

        grad_norm.append(sum(iter_grad)/len(iter_grad))

        # Print epoch statistics
        print('Epoch [{}/{}], Training Loss: {:.4f}, Training Acc: {:.4f}, Validation Loss: {:.4f}, Validation Acc: {:.4f}'.format(epoch+1, epochs, epoch_loss, train_acc, val_loss, val_acc))

        with open(log_filename, "w") as f:
            sys.stdout = f
            print('Epoch [{}/{}], Training Loss: {:.4f}, Training Acc: {:.4f}, Validation Loss: {:.4f}, Validation Acc: {:.4f}'.format(epoch+1, epochs, epoch_loss, train_acc, val_loss, val_acc))
            sys.stdout = sys.__stdout__


    # Test Accuracy
    test_loss, test_acc = test(model, loss_fn, test_set, embedder)
    print('Testing Loss: {:.4f}, Testing Acc: {:.4f}'.format(test_loss, test_acc))

    # Record Total CPU Time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('CPU Time: ', elapsed_time)

    with open(log_filename, "w") as f:
        sys.stdout = f
        print('Testing Loss: {:.4f}, Testing Acc: {:.4f}'.format(test_loss, test_acc))
        print('CPU Time: ', elapsed_time)
        sys.stdout = sys.__stdout__

    return (train_acc_cumulative, val_acc_cumulative), (train_loss_cumulative, val_loss_cumulative), grad_norm



def test(model, loss_fn, dset, embedder):

    run_loss = 0.0

    accuracy = torchmetrics.Accuracy('binary').to(device)
    iter_acc = []

    # Iterate over batches of data
    for conjecture, step, labels in dset:

        with tr.no_grad():
            # conjecture = encode(conjecture, 256, embedder)
            # step = encode(step, 256, embedder)
            conjecture = embedder(conjecture)
            step = embedder(step)

            outputs = model(conjecture, step)

            labels = labels.to(device)
            loss = loss_fn(outputs.to(device), labels)

            # Update run loss
            run_loss += loss.item() * step.size(0)
            label_simpl = tr.where(labels[0] > labels[1], tr.tensor([1]).to(device), tr.tensor([0]).to(device))
            output_result = tr.where(outputs[0] > outputs[1], tr.tensor([1]).to(device), tr.tensor([0]).to(device))

            test_acc = accuracy(label_simpl, output_result)
            iter_acc.append(test_acc)

    test_loss = run_loss / len(dset)
    test_acc = sum(iter_acc) / len(iter_acc)  

    return test_loss, test_acc


if __name__ == '__main__':
    print("GPU ready = ", tr.cuda.is_available())

    net_acc = []
    net_loss = []
    net_grad = []

    if not os.path.exists('output'):
        os.makedirs('output')
    plot_idx = 1
    while os.path.exists(f'output/my_plot ({plot_idx})_1.png'):
        plot_idx += 1
    plot_filename = f'output/my_plot ({plot_idx})'

    log_filename_idx = 1
    while os.path.exists(f"output/logs_{log_filename_idx}.txt"):
        log_filename_idx += 1
    log_filename = f"output/logs_{log_filename_idx}.txt"
    
    for i in range(3):

        model = mdl.SiameseTransformer(256, nhead=32)
        # model = mdl.SiameseCNNLSTM(256, 256)

        model.to(device)
        acc, loss, grad = train(model, lr=0.001, epochs=4, data_size=10, write_log=True, log_filename=log_filename)

        net_acc.append(acc)
        net_loss.append(loss)
        net_grad.append(grad)


    # Graph the accuracy curves
    fig, ax = plt.subplots()
    ax.plot(net_acc[0][0], color='blue', label='Train 1')
    ax.plot(net_acc[1][0], linestyle='--', color='blue', label='Train 2')
    ax.plot(net_acc[2][0], linestyle=':', color='blue', label='Train 3')
    ax.plot(net_acc[0][1], color='red', label='Validation 1')
    ax.plot(net_acc[1][1], linestyle='--', color='red', label='Validation 2')
    ax.plot(net_acc[2][1], linestyle=':', color='red', label='Validation 3')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Siamese Transformer 32 Attention Heads')        # <- TODO: Set title to model used
    ax.legend()

    fig.savefig(f'{plot_filename}_1.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # Graph the loss curves
    fig, ax = plt.subplots()
    ax.plot(net_loss[0][0], color='blue', label='Train 1')
    ax.plot(net_loss[1][0], linestyle='--', color='blue', label='Train 2')
    ax.plot(net_loss[2][0], linestyle=':', color='blue', label='Train 3')
    ax.plot(net_loss[0][1], color='red', label='Validation 1')
    ax.plot(net_loss[1][1], linestyle='--', color='red', label='Validation 2')
    ax.plot(net_loss[2][1], linestyle=':', color='red', label='Validation 3')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Siamese Transformer 32 Attention Heads')        # <- TODO: Set title to model used
    ax.legend()

    fig.savefig(f'{plot_filename}_2.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # Graph grad l1 norm
    fig, ax = plt.subplots()
    ax.plot(net_grad[0], label='Iteration 1')
    ax.plot(net_grad[1], label='Iteration 2')
    ax.plot(net_grad[2], label='Iteration 3')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Gradient L1 Norm')
    ax.set_title('Siamese Transformer 32 Attention Heads')        # <- TODO: Set title to model used
    ax.legend()

    # randomly generate a name to save the 3 plots
    fig.savefig(f'{plot_filename}_3.png', dpi=300, bbox_inches='tight', pad_inches=0.1)