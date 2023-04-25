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

# TODO: Decide if there is a benefit to using Wandb for logging

def train(model, epochs=50, data_size=1000, lr=0.001):

    # TODO: Determine if there is a better loss function and optimizer for our task
    loss_fn = tr.nn.CrossEntropyLoss()
    optimizer = tr.optim.SGD(model.parameters(), lr=lr)

    # TODO: Use parser + dataset builder to import dataset here
    data_loader = None

    # Setup metrics
    train_loss_cumulative = []        # Track the loss to graph
    test_loss_cumulative = []


    flops, params = get_model_complexity_info(model, data_loader, print_per_layer_stat=False)   # Track FLOPs
    print('FLOPs: ', flops, ', Total Parameters: ', params)

    start_time = time.time()    # Track CPU time

    # Run the training loop
    for epoch in range(epochs):

        run_loss = 0.0
        
        # Iterate over batches of data
        for conjecture, step, labels in data_loader:

            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(conjecture, step)
            loss = loss_fn(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update run loss
            run_loss += loss.item() * step.size(0)

        # Run Test Loss
        test_loss = test(model)

        # Update cumulative loss
        train_loss_cumulative.append(epoch_loss)
        test_loss_cumulative.append(test_loss)

        # Print epoch statistics
        epoch_loss = run_loss / len(data_size)
        print('Epoch [{}/{}], Training Loss: {:.4f}, Testing Loss: {:.4f}'.format(epoch+1, epochs, epoch_loss, test_loss))


    # Record Total Number of FLOPs
    total_flops, total_params = get_model_complexity_info(model, data_loader, print_per_layer_stat=False)
    print('Total FLOPs: ', total_flops, ', Total Parameters: ', total_params)

    # Record Total CPU Time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('CPU Time: ', elapsed_time)

    # Graph the loss curves
    plt.plot(train_loss_cumulative, 'b-')
    plt.plot(test_loss_cumulative, 'r-')
    plt.plot()
    plt.legend(["Train", "Test"])
    plt.show()



def test(model, data_size=1000):

    loss_fn = tr.nn.CrossEntropyLoss()

    # TODO: Use parser + dataset builder to import dataset here
    data_loader = None

    run_loss = 0.0

    # Iterate over batches of data
    for conjecture, step, labels in data_loader:

        with tr.no_grad():

            outputs = model(conjecture, step)
            loss = loss_fn(outputs, labels)

            # Update run loss
            run_loss += loss.item() * step.size(0)

    test_loss = run_loss / len(data_size)

    return test_loss