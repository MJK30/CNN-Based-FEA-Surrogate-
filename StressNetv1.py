import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


image_dir = "data/stress_images"
csv_path = "data/max_stress.csv"
model_path = "models/cnn_model.pt"
os.makedirs("models", exist_ok = True)


class StressImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = Image.open(os.path.join(self.image_dir, row['image_file'])).convert("L")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(float(row['max_stress']), dtype=torch.float32)
        return image, label

transform = transforms.Compose([
    transforms.Resize([256,256]),
    transforms.ToTensor()
                            ])

df = pd.read_csv(csv_path)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_set = StressImageDataset(train_df, image_dir, transform)
val_set = StressImageDataset(val_df, image_dir, transform)
test_set = StressImageDataset(test_df, image_dir, transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle= True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

# define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        # input [1,256, 256], output [16, 128, 128]
        nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        # input [16, 128, 128], output [32, 64, 64]
        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        # input [32, 64, 64], output [64, 32, 32]
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        # flatten the Convolution Layers
        nn.Flatten(),
        # input [64 * 32 * 32], ouput 128
        nn.Linear(64*32*32, 128), nn.ReLU(),
        # input 128, ouput 1
        nn.Linear(128, 1)
        )
    
    def forward(self,x):
        return self.net(x).squeeze()
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# intializes the model
model = CNNModel().to(device= device)
# adaptive Gradient Based Optimizer
optimizer = optim.Adam(model.parameters(), lr= 0.001)
# Mean Square Error as the loss function
criterion = nn.MSELoss()


for epoch in range(10):
    model.train()
    train_loss = 0
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)
        optimizer.zero_grad()  # clears the previous gradient
        output = model(img)  # prediction from CNN
        loss = criterion(output, label)  # compute MSE prediction bw prediction and true value
        loss.backward()  # backpropagates gradients through the model
        optimizer.step()  # update the model weights
        train_loss += loss.item()  # update total training loss
        
    model.eval()  # switch to evaluation mode for validation 
    val_loss = 0
    with torch.no_grad():  # store no gradients for validation
        for img, label in val_loader:
            img, label = img.to(device), label.to(device)
            ouput = model(img)
            val_loss += criterion(ouput, label).item()  # compute val_loss with backpropagation
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")
    
torch.save(model.state_dict(), model_path)   


model.eval()
pred, targets = [], []
with torch.no_grad():
    for img, label in test_loader:
        img = img.to(device)
        output = model(img).cpu().numpy()
        pred.extend(output)
        targets.extend(label.numpy())
        
# Mean Absolute Error - Avg size of the error (Smaller is better)
mae = mean_absolute_error(targets, pred)
# Mean Squared Error - Avg size of squared error. Larger error gets penalized more (samller is better)
mse = mean_squared_error(targets, pred)
rmse = mse ** 0.5
# R2 Score - variance in the true output (Higher is better)
r2 = r2_score(targets, pred)

print(f"\n📊 Evaluation on Test Set:")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.3f}")
