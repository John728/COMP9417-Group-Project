import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

def binary_accuracy(output, target):
    with torch.no_grad():
        if target.dtype != torch.bool:
            target = target.bool()
        pred = torch.sigmoid(output) > 0.5
        pred = pred.squeeze(1)
        assert pred.shape[0] == len(target)
        correct = torch.sum(pred == target)
    return correct.item() / len(target)

class MultiTaskNN(nn.Module):
    def __init__(self, input_size, hidden_size, task_output_sizes):
        super(MultiTaskNN, self).__init__()
        self.shared_layer = nn.Linear(input_size, hidden_size)
        self.task_layers = nn.ModuleList([
            nn.Linear(hidden_size, output_size) for output_size in task_output_sizes
        ])
        self.relu = nn.ReLU()

    def forward(self, x):
        shared_output = self.relu(self.shared_layer(x))
        task_outputs = [task_layer(shared_output) for task_layer in self.task_layers]
        return task_outputs

def l21_regularization(model, reg_strength):
    reg_loss = 0.0
    for param in model.parameters():
        if param.dim() > 1:
            l2_norm = torch.sqrt(torch.sum(param ** 2, dim=1)).sum()
        else:
            l2_norm = torch.sqrt(torch.sum(param ** 2))
        reg_loss += l2_norm
    return reg_strength * reg_loss

input_size = 111  
hidden_size = 500
task_output_sizes = [1] * 11
model = MultiTaskNN(input_size, hidden_size, task_output_sizes)
criterion = nn.BCEWithLogitsLoss()
X_train_normalized = np.load('X_train_normalized.npy')
y_train = np.load('y_train.npy')
X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
y_train_tensors = [torch.tensor(y_train[:, i], dtype=torch.float32) for i in range(y_train.shape[1])]

optimizer = optim.Adam([
    {'params': model.shared_layer.parameters()},
    {'params': model.task_layers.parameters()}
], lr=0.001)
reg_strength = 0.0001

for epoch in range(200):
    optimizer.zero_grad()
    y_preds = model(X_train_tensor)
    loss = 0
    for i, y_pred_i in enumerate(y_preds):
        y_true_i = y_train_tensors[i].view(-1, 1)
        loss += criterion(y_pred_i, y_true_i)
    reg_loss = l21_regularization(model, reg_strength)
    total_loss = loss + reg_loss
    total_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss.item()}')


model.eval()
accuracies = []
cross_entropy_losses = []
with torch.no_grad():
    y_preds_train = model(X_train_tensor)
    for i, y_pred_i in enumerate(y_preds_train):
        accuracy = binary_accuracy(y_pred_i, y_train_tensors[i])
        accuracies.append(accuracy)
        print(f'Task {i+1} Accuracy: {accuracy:.4f}')
    
        y_true_i = y_train_tensors[i].view(-1, 1)
        task_loss = criterion(y_pred_i, y_true_i)
        cross_entropy_losses.append(task_loss.item())
        print(f'Task {i+1} Cross-Entropy Loss: {task_loss.item()}')

average_accuracy = sum(accuracies) / len(accuracies)
print(f'Average Accuracy: {average_accuracy:.4f}')



X_test_normalized = np.load('X_test_normalized.npy')
X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
model.eval() 
with torch.no_grad():
    y_preds_test = model(X_test_tensor)

binary_outputs = [torch.sigmoid(y_pred).numpy() > 0.5 for y_pred in y_preds_test]

test_outputs_df = pd.DataFrame({
    f'Task {i+1} Predictions': output.astype(int).flatten()
    for i, output in enumerate(binary_outputs)
})

excel_path = 'y_test.csv'
test_outputs_df.to_csv(excel_path, index=False, header = False)

print(f'Predictions saved to {excel_path}')
