# COMOP9417 Naive Approach to MTL using LogisiticRegression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

X_train = np.load('X_train_filled.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test_filled.npy')

X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

models = []
val_losses = []

for i in range(y_train.shape[1]):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_part, y_train_part[:, i])
    models.append(model)
    y_pred = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val[:, i], y_pred)
    val_losses.append(loss)
    print(f"Loss for task {i+1}: {loss}")

y_test_pred = models[0].predict_proba(X_test)[:, 1]

tasks = [f'Task {i+1}' for i in range(len(val_losses))]
total = sum(val_losses)
print("Total loss:", total)
plt.figure(figsize=(10, 6))
plt.bar(tasks, val_losses, color='skyblue')
plt.xlabel('Tasks')
plt.ylabel('Loss')
plt.title('Validation Loss for Each Task')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()