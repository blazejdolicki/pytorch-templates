import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(num_hidden, num_outputs)

    def forward(self, x):
        print("x type", x.dtype)
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x

if __name__ == "__main__":
    num_inputs = 10
    num_hidden = 40
    num_outputs = 7
    batch_size = 8

    x = torch.randn(batch_size, num_inputs)
    print("x dtype", x.dtype)
    model = Net(num_inputs, num_hidden, num_outputs)
    outputs = model(x)

    print("Model")
    print(model)
    print("Outputs")
    print(outputs)
