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
import matplotlib.pyplot as plt

# StressNetv1 models has high RMSE and MSE values, indicating large errors in prediction
# StressNetv2 tries to tackle the problem by making more tuned and receptive
# aimed to improve feature selection and make the model generalize better

# Increased the layer of Convolutional blocks
# added BatchNorm2d after each layer to Normalize and stabilise
# added random dropouts for Convoluted layers as well, helps the model to train better
# LeakyReLU used to have more active neurons
# data augmentation during training
# changes the loss function from MSE to Huber Loss - Detected outliers (RMSE>>MSE). Huber Loss more robust to Outliers
# Normalizing the Von Mises Stress from the Stress Dataset - for better prediction
# Used MC simulation instead of standard evaluation() of test dataset



image_dir = "data/stress_images"
csv_path = "data/max_stress.csv"
model_path = "models/cnn_modelv2.pt"
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
        label = torch.tensor(float(row['max_stress_norm']), dtype=torch.float32)
        return image, label

# introducing data augmentation to improve the training models
transform_train = transforms.Compose([
    transforms.RandomResizedCrop((128, 128), scale=(0.9, 1.1), ratio=(0.9, 1.1)), # introduces scaling and zooming
    transforms.RandomRotation(degrees=10), # simulates random rotations
    transforms.RandomHorizontalFlip(p=0.5), # mirroring
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # adds small skews
    transforms.ToTensor()
                            ])
transform_val = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

df = pd.read_csv(csv_path)
max_stress_value = df["max_stress"].max()
df["max_stress_norm"] = df["max_stress"] / max_stress_value # normalizing the stress values for better prediction

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

train_set = StressImageDataset(train_df, image_dir, transform_train)
val_set = StressImageDataset(val_df, image_dir, transform_val)
test_set = StressImageDataset(test_df, image_dir, transform_train)

train_loader = DataLoader(train_set, batch_size=32, shuffle= True)
val_loader = DataLoader(val_set, batch_size=32)
test_loader = DataLoader(test_set, batch_size=32)

# define the CNN Model
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
        # input [1,128, 128], output [32, 64, 64]
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.1),  # random masking of 10% data for robust training
        nn.MaxPool2d(2),
        
        # input [32, 64, 64], output [64, 32, 32]
        nn.Conv2d(32, 64, kernel_size=3, padding=1), 
        nn.BatchNorm2d(64),
        nn.LeakyReLU(), 
        nn.Dropout2d(p=0.1),  # random masking of 10% data for robust training
        nn.MaxPool2d(2),
        
        
        # input [64, 32, 32], output [128, 16, 16]
        nn.Conv2d(64, 128, kernel_size=3, padding=1), 
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.1),  # random masking of 10% data for robust training
        nn.MaxPool2d(2),
        
        # input [128, 16, 16], output [256, 8, 8]
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.1),  # random masking of 10% data for robust training
        nn.MaxPool2d(2),

        # flatten the Convolution Layers
        nn.Flatten(),
        
        # input [256 * 8 * 8], ouput 128
        nn.Linear(256*8*8, 128),
        nn.Dropout(0.3),
        nn.ReLU(),
        
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
criterion = nn.HuberLoss(delta=1)


for epoch in range(30):
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


def mc_predict(model, inputs, T=30):
    """Runs a Monte Carlo Simulation
    MC analysis tests through train() -> all dropouts considered giving a network with different masking everytime for evaluation 
    

    Args:
        model (): CNN Model
        inputs (): Image
        T: Defaults to 30.

    Returns:
        mean and variance after T runs of simulation
    """
    model.train()
    preds = []
    
    for _ in range(T):
        preds.append(model(inputs).unsqueeze(0))
    preds = torch.cat(preds, dim=0)
    mean = preds.mean(dim=0)
    var = preds.var(dim=0)
    return mean, var

model.eval()
all_means, all_vars, all_targets = [], [], []


# Inference not through eval(), which will ignore all the dropout() and give a deterministic answer
# Uncertainity Quantification through Monte Carlo Analysis
with torch.no_grad():
    for img, label in test_loader:
        img = img.to(device)
        mean, var = mc_predict(model, img, T=30) # Monte Carlo Simulation used to get a mean and average of T Runs
        all_means.append(mean.cpu())
        all_vars.append(var.cpu())
        all_targets.append(label)

all_means   = torch.cat(all_means).numpy() * max_stress_value
all_vars    = torch.cat(all_vars).numpy() * (max_stress_value**2)
all_targets = torch.cat(all_targets).numpy() * max_stress_value
        
# Mean Absolute Error - Avg size of the error (Smaller is better)
mae = mean_absolute_error(all_targets, all_means)
# Mean Squared Error - Avg size of squared error. Larger error gets penalized more (samller is better)
mse = mean_squared_error(all_targets, all_means)
rmse = mse ** 0.5
# R2 Score - variance in the true output (Higher is better)
r2 = r2_score(all_targets, all_means)

print(f"\n Evaluation on Test Set:")
print(f"MAE  = {mae:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"R²   = {r2:.3f}")
