"""
The three recurrent models and their shared PyTorch training loop.
"""
from copy import deepcopy

import torch
from torch import nn

LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001
PATIENCE = 20

torch.set_num_threads(1)  # The models are small and run on the CPU.


class RNN(nn.Module):
    def __init__(self, kind, hidden_size, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.kind = kind
        self.hidden_size = hidden_size

        # Linear layers supply weights and biases with PyTorch's default initializstion.
        self.input_layer = nn.Linear(1, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        if kind in ('Elman', 'Multi'):
            self.hidden_feedback = nn.Linear(hidden_size, hidden_size, bias=False)
        if kind in ('Jordan', 'Multi'):
            self.output_feedback = nn.Linear(1, hidden_size, bias=False)

    def forward(self, x):
        """Each row of x is a window of past values; predict the next value."""
        hidden = torch.zeros(len(x), self.hidden_size)
        output = torch.zeros(len(x), 1)
        context = torch.zeros(len(x), 1)

        for step in range(x.shape[1]):
            context = 0.5 * context + 0.5 * output
            total = self.input_layer(x[:, step:step + 1])
            if self.kind in ('Elman', 'Multi'):
                total = total + self.hidden_feedback(hidden)
            if self.kind in ('Jordan', 'Multi'):
                total = total + self.output_feedback(context)
            hidden = torch.tanh(total)
            output = self.output_layer(hidden)

        # Only the output at the end of the window is compared with a target.
        return output.squeeze(1)

    def predict(self, x):
        """Return NumPy values so that the experiment can calculate its errors."""
        with torch.no_grad():
            return self(torch.tensor(x, dtype=torch.float32)).numpy()


def train(model, x, y, epochs=200, validation=None):
    """Use validation for early stopping or train for a fixed number of epochs."""
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    if validation is not None:
        x_val = torch.tensor(validation[0], dtype=torch.float32)
        y_val = torch.tensor(validation[1], dtype=torch.float32)

    # Adam's weight decay penalizes all parameters (including biases)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    loss_function = nn.MSELoss()
    history = []
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss = loss_function(model(x), y)
        loss.backward()  # PyTorch differentiates through the recurrence.
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            training_loss = loss_function(model(x), y).item()
            validation_loss = (loss_function(model(x_val), y_val).item()
                               if validation is not None else training_loss)
        history.append((epoch, training_loss, validation_loss))

        if validation is not None:
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_weights = deepcopy(model.state_dict())
            if epoch - best_epoch >= PATIENCE:
                break

    if validation is not None:
        model.load_state_dict(best_weights)
        return best_epoch, history
    return epochs, history
