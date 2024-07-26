from torch import nn


# generates the output of the transformer-regression model (with at lest one activation function and two linear layers)
class generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(generator, self).__init__()

        self.activation = nn.ReLU()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, output_size)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x
