"""
Installation Packages:
  - PyTorch
  - MatPlotLib
  - PTFLOPs: https://pypi.org/project/ptflops/
"""

import torch as tr
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
import time
import dataset as ds
import model as mdl
import sklearn.metrics as metrics
import torchmetrics
from thop import profile
import util

device = tr.device("cuda")

# TODO: Decide if there is a benefit to using Wandb for logging

def train(model, epochs=50, data_size=1000, lr=0.01, loss_fn=None, optimizer=None):

    # Set default loss and optimizer
    if loss_fn == None: loss_fn = tr.nn.CrossEntropyLoss()
    if optimizer == None: optimizer = tr.optim.SGD(model.parameters(), lr=lr)

    accuracy = torchmetrics.Accuracy('binary').to(device)


    print("Generating dataset...")
    train_set, val_set, test_set = ds.dataset_builder(data_size)

    # Setup metrics
    train_loss_cumulative = []        # Track the loss to graph
    val_loss_cumulative = []
    train_acc_cumulative = []
    val_acc_cumulative = []

    start_time = time.time()    # Track CPU time

    print("Begining training loop...")
    # Run the training loop
    for epoch in range(epochs):

        run_loss = 0.0
        iter_acc = []
        
        idx = 0
        # Iterate over batches of data
        for conjecture, step, labels in train_set:

            # Zero the gradients
            optimizer.zero_grad()
        
            conjecture = conjecture.squeeze(dim=1).to(device)
            step = step.squeeze(dim=1).to(device)
            con_label = tr.stack([labels]*conjecture.size()[1]).permute(1, 0, 2).to(device)
            step_label = tr.stack([labels]*step.size()[1]).permute(1, 0, 2).to(device)

            if epoch == 0 and idx == 0:
                flops, params = profile(model, inputs=(conjecture, step, con_label, step_label), verbose=False)
                print("==============================================================")
                print(f"  FLOPs Per Batch: {int(flops)}, Model Parameters: {int(params)}")
                print("==============================================================")

            # Forward pass
            outputs = model(conjecture, step, con_label, step_label)
            labels = labels[:, -2:].to(device)
            labels[:, 1] = 1 - labels[:, 1]
            loss = loss_fn(outputs.to(device), labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update run loss
            run_loss += loss.item() * step.size(0)

            # Update Accuracy
            label_simpl = tr.where(labels[:, 0] > labels[:, 1], tr.tensor([1]).to(device), tr.tensor([0]).to(device))
            output_result = tr.where(outputs[:, 0] > outputs[:, 1], tr.tensor([1]).to(device), tr.tensor([0]).to(device))
            train_acc = accuracy(label_simpl, output_result)
            iter_acc.append(train_acc)
            idx += 1

        # Run Test Loss
        val_loss, val_acc = test(model, loss_fn, val_set)
        epoch_loss = run_loss / len(train_set)
        train_acc = sum(iter_acc) / len(iter_acc)        

        # Update cumulative loss
        train_loss_cumulative.append(epoch_loss)
        val_loss_cumulative.append(val_loss)

        train_acc_cumulative.append(train_acc.cpu())
        val_acc_cumulative.append(val_acc.cpu())

        # Print epoch statistics
        print('Epoch [{}/{}], Training Loss: {:.4f}, Training Acc: {:.4f}, Validation Loss: {:.4f}, Validation Acc: {:.4f}'.format(epoch+1, epochs, epoch_loss, train_acc, val_loss, val_acc))


    # Test Accuracy
    test_loss, test_acc = test(model, loss_fn, test_set)
    print('Testing Loss: {:.4f}, Testing Acc: {:.4f}'.format(test_loss, test_acc))

    # Record Total CPU Time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('CPU Time: ', elapsed_time)


    return (train_acc_cumulative, val_acc_cumulative), (train_loss_cumulative, val_loss_cumulative)



def test(model, loss_fn, dset):

    run_loss = 0.0

    accuracy = torchmetrics.Accuracy('binary').to(device)
    iter_acc = []

    # Iterate over batches of data
    for conjecture, step, labels in dset:

        with tr.no_grad():
            conjecture = conjecture.squeeze(dim=1).to(device)
            step = step.squeeze(dim=1).to(device)
            con_label = tr.stack([labels]*conjecture.size()[1]).permute(1, 0, 2).to(device)
            step_label = tr.stack([labels]*step.size()[1]).permute(1, 0, 2).to(device)

            outputs = model(conjecture, step, con_label, step_label)

            labels = labels[:, -2:].to(device)
            labels[:, 1] = 1 - labels[:, 1]
            loss = loss_fn(outputs.to(device), labels)

            # Update run loss
            run_loss += loss.item() * step.size(0)
            label_simpl = tr.where(labels[:, 0] > labels[:, 1], tr.tensor([1]).to(device), tr.tensor([0]).to(device))
            output_result = tr.where(outputs[:, 0] > outputs[:, 1], tr.tensor([1]).to(device), tr.tensor([0]).to(device))

            test_acc = accuracy(label_simpl, output_result)
            iter_acc.append(test_acc)

    test_loss = run_loss / len(dset)
    test_acc = sum(iter_acc) / len(iter_acc)  

    return test_loss, test_acc


if __name__ == '__main__':
    print("GPU ready = ", tr.cuda.is_available())

    net_acc = []
    net_loss = []
    
    for i in range(3):

        model = mdl.SiameseTransformer(256)
        # model = mdl.SiameseCNNLSTM(256, 256)

        model.to(device)
        acc, loss = train(model, data_size=500, epochs=5, lr=0.1)

        net_acc.append(acc)
        net_loss.append(loss)
    

    acc_averages = util.list_average(net_acc)
    loss_averages = util.list_average(net_loss)


    # Graph the accuracy curves
    fig, ax = plt.subplots()
    ax.plot(acc_averages[0], label='Train')
    ax.plot(acc_averages[1], label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Siamese Transformer')        # <- TODO: Set title to model used
    ax.legend()

    plt.show()

    # Graph the loss curves
    fig, ax = plt.subplots()
    ax.plot(loss_averages[0], label='Train')
    ax.plot(loss_averages[1], label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Siamese Transformer')        # <- TODO: Set title to model used
    ax.legend()

    plt.show()