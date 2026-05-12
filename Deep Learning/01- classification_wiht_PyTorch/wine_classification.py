import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import wandb
import sklearn

# ───  Dataset ───────────────────────────────────────────────
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    print(df.info())
    return df

class CustomDataset(Dataset):
    def __init__(self, X, y): 
        self.inputs = torch.tensor(X, dtype=torch.float32)
        self.labels = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        data = self.inputs[idx]
        label = self.labels[idx]
        return {'input': data, 'label': label}
    
def get_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = False):
    return DataLoader(dataset, batch_size, shuffle=shuffle)

def split_dataset(dataset, split_ratio):
    dataset_size = len(dataset)
    split_size = int(dataset_size * split_ratio)
    train_dataset, eval_dataset = random_split(dataset, [split_size, dataset_size - split_size])
    return train_dataset, eval_dataset


# ─── Modeling ──────────────────────────────────────────────
class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, input):
        output = self.linear(input)
        return output
    
class LinearModelWithActivation(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, input):
        return self.network(input)


# ─── Training & Evaluation ─────────────────────────────────
def train_epoch(model, loss_fn, optimizer, dataloader):
    model.train()
    loss_history = []
    accuracy_history = []

    for batch in dataloader:
        inputs = batch["input"]
        labels = batch["label"]

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, dim=1)
        accuracy = (predicted == labels).float().mean().item()

        loss_history.append(loss.item())
        accuracy_history.append(accuracy)

    return sum(loss_history) / len(loss_history), sum(accuracy_history) / len(accuracy_history)


def eval_epoch(model, loss_fn, dataloader):
    model.eval()
    loss_history = []
    accuracy_history = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"]
            labels = batch["label"]

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            predicted = torch.argmax(outputs, dim=1)
            accuracy = (predicted == labels).float().mean().item()

            loss_history.append(loss.item())
            accuracy_history.append(accuracy)

    return sum(loss_history) / len(loss_history), sum(accuracy_history) / len(accuracy_history)


# ─── Main ──────────────────────────────────────────────────
def main():
    # Hyperparameters
    N_EPOCHS = 100
    IN_DIM = 13 
    OUT_DIM = 3
    B_SIZE = 4
    LR = 0.005
    SEED = 42

    torch.manual_seed(SEED)

    df = load_data("/content/wine.csv")
    
    # Split DataFrame avant scaling
    X = df.drop('Wine', axis=1).values
    y = (df['Wine'].values - 1)
    
    split_idx = int(0.8 * len(X))
    X_train, X_eval = X[:split_idx], X[split_idx:]
    y_train, y_eval = y[:split_idx], y[split_idx:]

    # Normalisation (fit sur train uniquement)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_eval = scaler.transform(X_eval)

    # Datasets
    train_dataset = CustomDataset(X_train, y_train)
    eval_dataset = CustomDataset(X_eval, y_eval)

    train_dataloader = get_dataloader(dataset=train_dataset, batch_size=B_SIZE, shuffle=True)
    eval_dataloader = get_dataloader(dataset=eval_dataset, batch_size=B_SIZE, shuffle=False)

    model_with_sgd = LinearModelWithActivation(in_dim=IN_DIM, out_dim=OUT_DIM)
    model_with_adam = LinearModelWithActivation(in_dim=IN_DIM, out_dim=OUT_DIM)

    sgd_optimizer = optim.SGD(model_with_sgd.parameters(), lr=LR) 
    adam_optimizer = optim.Adam(model_with_adam.parameters(), lr=LR)

    myloss_fn = nn.CrossEntropyLoss()

    for optimizer_name, model, optimizer in [
        ("SGD", model_with_sgd, sgd_optimizer),
        ("Adam", model_with_adam, adam_optimizer)
    ]:
        wandb.init(
            project="wine-classification",
            name=f"run_{optimizer_name}",
            config={
                "epochs": N_EPOCHS,
                "batch_size": B_SIZE,
                "learning_rate": LR,
                "optimizer": optimizer_name,
                "normalized": True,
                "activation": "sigmoid",
                "hidden_dim": 32
            }
        )

        print(f"\n{'='*50}")
        print(f"Training with {optimizer_name} optimizer (normalized):")
        print(f"{'='*50}")

        for epoch in range(N_EPOCHS):
            train_loss, train_acc = train_epoch(
                model=model, loss_fn=myloss_fn,
                optimizer=optimizer, dataloader=train_dataloader)
            eval_loss, eval_acc = eval_epoch(
                model=model, loss_fn=myloss_fn,
                dataloader=eval_dataloader)
            print(f"Epoch {epoch+1}/{N_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.4f}")
            wandb.log({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "eval_loss": eval_loss,
                "eval_acc": eval_acc
            })
        wandb.finish()

if __name__ == "__main__":
    main()