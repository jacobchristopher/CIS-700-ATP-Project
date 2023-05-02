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

device = tr.device("cuda")

# TODO: Decide if there is a benefit to using Wandb for logging

def train(model, epochs=50, data_size=1000, lr=0.01, loss_fn=None, optimizer=None):

    # Set default loss and optimizer
    if loss_fn == None: loss_fn = tr.nn.CrossEntropyLoss()
    if optimizer == None: optimizer = tr.optim.SGD(model.parameters(), lr=lr)


    print("Generating dataset...")
    train_set, val_set, test_set = ds.dataset_builder(data_size)

    # Setup metrics
    train_loss_cumulative = []        # Track the loss to graph
    test_loss_cumulative = []


    # FIXME
    # flops, params = get_model_complexity_info(model, train_set, print_per_layer_stat=False)   # Track FLOPs
    # print('FLOPs: ', flops, ', Total Parameters: ', params)

    start_time = time.time()    # Track CPU time

    print("Begining training loop...")
    # Run the training loop
    for epoch in range(epochs):

        run_loss = 0.0
        
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
            loss = loss_fn(outputs.to(device), labels[:, :2].to(tr.long).to(device))
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update run loss
            run_loss += loss.item() * step.size(0)

        # Run Test Loss
        test_loss = test(model, loss_fn, val_set)
        epoch_loss = run_loss / len(train_set)

        # Update cumulative loss
        train_loss_cumulative.append(epoch_loss)
        test_loss_cumulative.append(test_loss)

        # Print epoch statistics
        print('Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch+1, epochs, epoch_loss, test_loss))


    # Test Accuracy
    test_loss = test(model, loss_fn, test_set)
    print('Testing Loss: {:.4f}'.format(test_loss))

    # FIXME
    # Record Total Number of FLOPs
    # total_flops, total_params = get_model_complexity_info(model, train_set, print_per_layer_stat=False)
    # print('Total FLOPs: ', total_flops, ', Total Parameters: ', total_params)

    # Record Total CPU Time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('CPU Time: ', elapsed_time)

    # Graph the loss curves
    plt.plot(train_loss_cumulative, 'b-')
    plt.plot(test_loss_cumulative, 'r-')
    plt.plot()
    plt.legend(["Train", "Validation"])
    plt.show()



def test(model, loss_fn, dset):

    run_loss = 0.0

    # Iterate over batches of data
    for conjecture, step, labels in dset:

        with tr.no_grad():
            conjecture = conjecture.squeeze(dim=1).to(device)
            step = step.squeeze(dim=1).to(device)
            con_label = tr.stack([labels]*conjecture.size()[1]).permute(1, 0, 2).to(device)
            step_label = tr.stack([labels]*step.size()[1]).permute(1, 0, 2).to(device)

            outputs = model(conjecture, step, con_label, step_label)
            loss = loss_fn(outputs.to(device), labels[:, :2].to(tr.long).to(device))

            # Update run loss
            run_loss += loss.item() * step.size(0)

    test_loss = run_loss / len(dset)

    return test_loss


if __name__ == '__main__':
    print("GPU ready = ", tr.cuda.is_available())

    model = mdl.SiameseTransformer(256)
    # model = mdl.SiameseCNNLSTM(256, 256)
    model.to(device)
    train(model, data_size=100)