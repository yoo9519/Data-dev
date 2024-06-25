# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch

print(torch.__version__)
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

# %%
import seaborn

data['rating'].plot(kind='hist')

# %%
columns = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv('/Users/yoo/pythonenv/datasets/ml-100k/u.data', sep='\t', names=columns)

data['user_id'] = data['user_id'] - 1
data['item_id'] = data['item_id'] - 1

train_data, test_data = train_test_split(data, test_size=0.2, random_state=9095)

train_tensor = torch.tensor(train_data.values, dtype=torch.long)
test_tensor = torch.tensor(test_data.values, dtype=torch.long)


# %%
# 데이터 구조
class MovieLensDataset(Dataset):
    def __init__(self, data):
        self.users = torch.tensor(data['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(data['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


train_dataset = MovieLensDataset(train_data)
test_dataset = MovieLensDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)


# 모델 아키텍처
class MatrixFactorization(pl.LightningModule):
    def __init__(self, num_users, num_items, embedding_size, lr=0.01):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
        self.lr = lr
        self.criterion = nn.MSELoss()

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        dot_product = (user_embeds * item_embeds).sum(dim=1)
        return dot_product

    def training_step(self, batch, batch_idx):
        users, items, ratings = batch
        outputs = self(users, items)
        loss = self.criterion(outputs, ratings)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        users, items, ratings = batch
        outputs = self(users, items)
        loss = self.criterion(outputs, ratings)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


num_users = data['user_id'].nunique()
num_items = data['item_id'].nunique()
embedding_size = 50

model = MatrixFactorization(num_users, num_items, embedding_size)

# %%
# 모델 학습 및 평가
trainer = pl.Trainer(max_epochs=20)
trainer.fit(model, train_loader, test_loader)


# %%
# 평가
def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    criterion = nn.MSELoss()

    for batch in test_loader:
        users, items, ratings = batch
        outputs = model(users, items)
        loss = criterion(outputs, ratings)
        test_loss += loss.item()

    avg_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_loss}")


evaluate(model, test_loader)


# 추천시스템 적용
def get_top_users_for_item(model, item_id, num_users, top_k=5):
    model.eval()
    users = torch.arange(num_users, dtype=torch.long)
    item_ids = torch.tensor([item_id] * num_users, dtype=torch.long)

    with torch.no_grad():
        predictions = model(users, item_ids)

    top_users = torch.argsort(predictions, descending=True)[:top_k]
    top_scores = predictions[top_users]
    return top_users, top_scores


item_id = 1  # 예측하고자 하는 item_id
top_k = 5

top_users, top_scores = get_top_users_for_item(model, num_items, num_users, top_k)

print(f"Top 5 users for item {num_items}: {top_users}")
print(f"Their predicted ratings: {top_scores}")