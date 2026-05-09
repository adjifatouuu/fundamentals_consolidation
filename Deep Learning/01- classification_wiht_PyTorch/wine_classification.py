import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import wandb


# ─── 1. Dataset ───────────────────────────────────────────────
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    print(df.info())
    return df

class CustomDataset(Dataset):
    def __init__(self, dataframe): 
        self.inputs = torch.tensor(dataframe.drop('Wine', axis=1).values, dtype=torch.float32)
        self.labels = torch.tensor((dataframe['Wine'].values - 1), dtype=torch.long)

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


# ─── 2. Modeling ──────────────────────────────────────────────
class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, input):
        output = self.linear(input)
        return output


# ─── 3. Training & Evaluation ─────────────────────────────────
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


# ─── 4. Main ──────────────────────────────────────────────────
def main():
    # Hyperparameters
    N_EPOCHS = 100
    IN_DIM = 13 
    OUT_DIM = 3
    B_SIZE = 4
    LR = 0.005
    SEED = 42

    torch.manual_seed(SEED)

    df = load_data("wine.csv")
    my_dataset = CustomDataset(dataframe=df)
    train_dataset, eval_dataset = split_dataset(my_dataset, split_ratio=0.8)

    train_dataloader = get_dataloader(dataset=train_dataset, batch_size=B_SIZE, shuffle=True)
    eval_dataloader = get_dataloader(dataset=eval_dataset, batch_size=B_SIZE, shuffle=False)

    model_with_sgd = LinearModel(in_dim=IN_DIM, out_dim=OUT_DIM)
    model_with_adam = LinearModel(in_dim=IN_DIM, out_dim=OUT_DIM)

    sgd_optimizer = optim.SGD(model_with_sgd.parameters(), lr=LR) 
    adam_optimizer = optim.Adam(model_with_adam.parameters(), lr=LR)

    myloss_fn = nn.CrossEntropyLoss()

    for optimizer_name, model, optimizer in [
        ("SGD", model_with_sgd, sgd_optimizer),
        ("Adam", model_with_adam, adam_optimizer)
    ]:
    
        wandb.init(
            project="wine-classification", 
            name="run_" + optimizer_name,
            config={
                "epochs": N_EPOCHS,
                "batch_size": B_SIZE,
                "learning_rate": LR,
                "optimizer": optimizer_name,
                "in_dim": IN_DIM,
                "out_dim": OUT_DIM
            }
        )

        print(f"\n{'='*50}")
        print(f"Training with {optimizer_name} optimizer:")
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