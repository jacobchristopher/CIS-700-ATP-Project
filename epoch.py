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


    # FIXME
    # flops, params = get_model_complexity_info(model, train_set, print_per_layer_stat=False)   # Track FLOPs
    # print('FLOPs: ', flops, ', Total Parameters: ', params)

    start_time = time.time()    # Track CPU time

    print("Begining training loop...")
    # Run the training loop
    for epoch in range(epochs):

        run_loss = 0.0
        iter_acc = []
        
        # Iterate over batches of data
        for conjecture, step, labels in train_set:

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            conjecture = conjecture.squeeze(dim=1).to(device)
            step = step.squeeze(dim=1).to(device)

            # print(conjecture.shape, step.shape)
            con_label = tr.stack([labels]*conjecture.size()[1]).permute(1, 0, 2).to(device)
            step_label = tr.stack([labels]*step.size()[1]).permute(1, 0, 2).to(device)

            # print(conjecture.shape, con_label.shape)
            outputs = model(conjecture, step, con_label, step_label)

            labels = labels[:, -2:].to(device)
            labels[:, 1] = 1 - labels[:, 1]
            loss = loss_fn(outputs.to(device), labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update run loss
            run_loss += loss.item() * step.size(0)

            label_simpl = tr.where(labels[:, 0] > labels[:, 1], tr.tensor([1]).to(device), tr.tensor([0]).to(device))
            output_result = tr.where(outputs[:, 0] > outputs[:, 1], tr.tensor([1]).to(device), tr.tensor([0]).to(device))

            train_acc = accuracy(label_simpl, output_result)
            iter_acc.append(train_acc)

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

    # FIXME
    # Record Total Number of FLOPs
    # total_flops, total_params = get_model_complexity_info(model, train_set, print_per_layer_stat=False)
    # print('Total FLOPs: ', total_flops, ', Total Parameters: ', total_params)

    # Record Total CPU Time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('CPU Time: ', elapsed_time)

    # Graph the loss curves
    fig, ax = plt.subplots()
    ax.plot(train_acc_cumulative, label='Train')
    ax.plot(val_acc_cumulative, label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Siamese Transformer')
    ax.legend()

    plt.show()



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

    model = mdl.SiameseTransformer(256)
    # model = mdl.SiameseCNNLSTM(256, 256)
    model.to(device)
    train(model, data_size=200, epochs=75)