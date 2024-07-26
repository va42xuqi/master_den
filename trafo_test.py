import torch
import math
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader


# Positional Encoding
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


# Define Transformer model
class SimpleTransformerModel(torch.nn.Module):
    def __init__(
        self,
        embedding_dim,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        dropout,
        prediction_len,
        output_size,
        num_players,
        num_features,
        categories_sizes,
    ):
        super(SimpleTransformerModel, self).__init__()
        self.transformer = torch.nn.Transformer(
            d_model=embedding_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.vs = VariableSelection(
            num_players, num_features, categories_sizes, embedding_dim
        )
        self.pos_encoder = PositionalEncoding(embedding_dims, max_allowed_seq_len)
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.fc_out = torch.nn.Linear(embedding_dim, prediction_len * output_size)
        self.prediction_len = prediction_len
        self.output_size = output_size

    def forward(self, src):
        src = self.vs(src)
        src = self.pos_encoder(src)
        src = self.dropout_layer(src)
        output = self.transformer(src, src)
        output = self.fc_out(output[:, -1, :])
        return output.view(output.size(0), self.prediction_len, self.output_size)


class VariableSelection(torch.nn.Module):
    def __init__(
        self,
        num_players,
        num_features,
        categories_sizes: list[int],
        embedding_dims,
    ):
        super().__init__()
        self.num_players = num_players
        self.num_features = num_features
        self.embedding_dims = embedding_dims
        # Embedding layer for each feature
        self.embeddings = torch.nn.ModuleList(
            [
                torch.nn.Embedding(categories_sizes[i], embedding_dims)
                for i in range(num_features)
            ]
        )
        # Linear layer to project the data to the desired dimensions
        self.linear = torch.nn.Linear(
            num_features * num_players * embedding_dims, embedding_dims
        )

    def forward(self, x):
        # Embedding for position and velocity (for simplicity, we use the same features for each player)
        y = (
            torch.zeros(
                (
                    x.size(0),
                    x.size(1),
                    self.num_players,
                    self.embedding_dims,
                    self.num_features,
                ),
            ).type_as(x)
            * 1.0
        )
        for i in range(self.num_features):
            y[..., i] = self.embeddings[i](x[..., i])

        # Squeeze the last two dimensions
        y = y.view(x.size(0), seq_len, -1)
        data = self.linear(y)
        return data


class LitModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Pass data through the Transformer model
        output = self(x)
        y = y[:, :, 0]

        # calculate loss
        criterion = torch.nn.MSELoss()
        loss = criterion(output, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Example usage
batch_size = 32
seq_len = 50
embedding_dims = 64
dropout = 0.1
max_allowed_seq_len = 1000
num_samples = 1000
num_players = 10
prediction_len = 100  # Number of future time steps to predict
# Output size for each prediction (e.g., pos_x, pos_y, vel_x, vel_y for each player)
output_size = 2

# Number of categories for each position and velocity coordinate
height_categories = int(15.24 * 100)  # Height categories (0 to 1524)
width_categories = int(28.65 * 100)  # Width categories (0 to 2865)
velocity_categories = int(11.11 * 100)  # Velocity categories (0 to 1111)

# Sample position and velocity data
x = torch.randint(0, width_categories, (num_samples, seq_len, num_players, 1))
y = torch.randint(0, height_categories, (num_samples, seq_len, num_players, 1))
v_x = torch.randint(0, velocity_categories, (num_samples, seq_len, num_players, 1))
v_y = torch.randint(0, velocity_categories, (num_samples, seq_len, num_players, 1))

# sample target position
target_pos_x = torch.randint(
    0, width_categories, (num_samples, prediction_len, num_players, 1)
) / float(width_categories)
target_pos_y = torch.randint(
    0, height_categories, (num_samples, prediction_len, num_players, 1)
) / float(height_categories)

# create the data
x = torch.cat((x, y, v_x, v_y), dim=-1)
y = torch.cat((target_pos_x, target_pos_y), dim=-1)


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, input, target):
        self.input = input
        self.target = target
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input_data = self.input[idx].to(self.device)
        target_data = self.target[idx].to(self.device)
        return input_data, target_data


dataset = CustomDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Initialize the Transformer model
transformer_model = SimpleTransformerModel(
    embedding_dim=embedding_dims,
    nhead=8,
    num_encoder_layers=6,
    dim_feedforward=2048,
    dropout=dropout,
    prediction_len=prediction_len,
    output_size=output_size,
    num_players=num_players,
    num_features=4,
    categories_sizes=[
        width_categories,
        height_categories,
        velocity_categories,
        velocity_categories,
    ],
)

# Initialize the Lightning model
lit_model = LitModel(transformer_model)

# Initialize the Lightning Trainer
trainer = pl.Trainer(max_epochs=10)

# Train the model
trainer.fit(lit_model, dataloader)
