import TransformerModel
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import DataCollection
import Transformer_config as config


def train_transformer(model, data_loader,val_loader, criterion, optimizer, num_epochs=config.num_epochs):
    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        total_train_loss = 0
        l = len(data_loader)
        for i, (inputs, labels, masks) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs, masks)  # マスクをモデルに渡す
            labels = labels.squeeze(1)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        print("評価モードに切り替え")
        with torch.no_grad():
            for inputs, labels, masks in val_loader:
                outputs = model(inputs, masks)  # マスクをモデルに渡す
                labels = labels.squeeze(1)
                loss = criterion(outputs, labels.long())
                total_val_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {total_train_loss / len(data_loader)}, Val Loss: {total_val_loss / len(val_loader)}')





model = TransformerModel.TransformerModel(config.input_dim, config.model_dim, config.num_heads, config.num_layers, config.num_classes, config.dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


data_class = DataCollection.DataCollection()
skeleton_train, skeleton_val, skeleton_test, labels_train, labels_val, labels_test, mask_train, mask_val, mask_test = data_class.get_dataset()

train_dataset = TensorDataset(skeleton_train, labels_train, mask_train)
train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)


val_dataset = TensorDataset(skeleton_val, labels_val, mask_val)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# モデル訓練の実行
train_transformer(model, train_data_loader, val_loader, criterion, optimizer)
torch.save(model.state_dict(), config.restore_model_path)
