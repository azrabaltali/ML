## Simple Regression
#     Make sure you add the bias feature to each training and test example.
#     Standardize the features using the mean and std computed over training data.

import sys
import numpy as np
from matplotlib import pyplot as plt
import scaling


# Read data matrix X and labels y from text file.
def read_data(file_name):
#  YOUR CODE HERE
  X = []
  y = []

  data = np.loadtxt(file_name)
  X = data[:,0]       #tüm satırların 0. sütunu yani metrekare
  y = data[:,1]       #tüm satırların 1. sütunu yani fiyat

  return X, y


# Implement gradient descent algorithm to compute w = [w0, w1].
def train(X, y, lamda, epochs):
#  YOUR CODE HERE

  m = len(y)
  w = np.zeros(X.shape[1])   # [w0, w1] başlangıçta 0
  J_history = []              # Maliyet geçmişi
    
  for epoch in range(epochs):
    # Gradyanı hesapla
    grad = compute_gradient(X, y, w)
        
    # w'yi güncelle
    w = w - lamda * grad
        
    # Maliyeti kaydet
    J = compute_cost(X, y, w)
    J_history.append(J)
    
  return w, J_history


# Compute Root mean squared error (RMSE)).
def compute_rmse(X, y, w):
#  YOUR CODE HERE
  m = len(y)
  y_pred = X @ w
  rmse = np.sqrt((1/m) * np.sum((y_pred - y)**2))
  return rmse


# Compute objective (cost) function.
def compute_cost(X, y, w):
#  YOUR CODE HERE
  m = len(y)
  y_pred = X @ w                              #maliyet fonksiyonunu hesaplıyoruz.
  J = (1/(2*m)) * np.sum((y_pred - y)**2)     #karesini alıyoruz çünkü hata negatif de olabilir pozitif de
  return J


# Compute gradient descent Algorithm.
def compute_gradient(X, y, w):
#  YOUR CODE HERE
    m = len(y)
    y_pred = X @ w
    grad = (1/m) * (X.T @ (y_pred - y))
    return grad



##======================= Main program =======================##

# Read the training and test data.
Xtrain, ttrain = read_data("train.txt")
Xtest, ttest = read_data("test.txt")

#  YOUR CODE HERE
# 1. Train verisinin mean/std'sini hesapla (2D olarak)
Xtrain_2d = Xtrain.reshape(-1, 1)
mean, std = scaling.mean_std(Xtrain_2d)
print(f"Mean: {mean}, Std: {std}")

# 2. Train ve test'i standardize et
Xtrain_norm = scaling.standardize(Xtrain_2d, mean, std).flatten()
Xtest_norm = scaling.standardize(Xtest.reshape(-1, 1), mean, std).flatten()

# 3. Bias sütunu ekle (1'ler)
Xtrain_b = np.c_[np.ones(len(Xtrain_norm)), Xtrain_norm]
Xtest_b = np.c_[np.ones(len(Xtest_norm)), Xtest_norm]

print(f"Xtrain_b shape: {Xtrain_b.shape}")  # (50, 2)
print(f"Xtest_b shape: {Xtest_b.shape}")    # (25, 2)

# 4. Gradient descent ile eğit
lamda = 0.1
epochs = 500
w, J_history = train(Xtrain_b, ttrain, lamda, epochs)

# 5. Parametreleri yazdır
print("\n=== Gradient Descent Sonucu ===")
print(f"w0 (bias): {w[0]:.4f}")
print(f"w1 (slope): {w[1]:.4f}")
print(f"Final Cost: {J_history[-1]:.4f}")

# 6. Normal denklemlerle çöz
w_normal = np.linalg.inv(Xtrain_b.T @ Xtrain_b) @ Xtrain_b.T @ ttrain
print("\n=== Normal Denklemler Sonucu ===")
print(f"w0 (bias): {w_normal[0]:.4f}")
print(f"w1 (slope): {w_normal[1]:.4f}")

# Karşılaştırma
print("\n=== Karşılaştırma ===")
print(f"Fark w0: {abs(w[0] - w_normal[0]):.6f}")
print(f"Fark w1: {abs(w[1] - w_normal[1]):.6f}")

# 7. RMSE hesapla
rmse_train = compute_rmse(Xtrain_b, ttrain, w)
rmse_test = compute_rmse(Xtest_b, ttest, w)
print(f"\n=== RMSE ===")
print(f"Train RMSE: {rmse_train:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")

# 8. J(w) vs Epoch grafiği
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), J_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('J(w)')
plt.title('Cost Function vs Epoch')
plt.grid(True)
plt.show()

# 9. Veri + Doğru grafiği
# Standardize edilmiş w'yi orijinal ölçeğe döndür
w1_orig = w[1] / std[0]
w0_orig = w[0] - w[1] * mean[0] / std[0]

plt.figure(figsize=(10, 6))
plt.scatter(Xtrain, ttrain, c='blue', marker='o', label='Train')
plt.scatter(Xtest, ttest, c='green', marker='x', label='Test')

# Doğru çizgisi
x_line = np.linspace(min(Xtrain.min(), Xtest.min()), 
                     max(Xtrain.max(), Xtest.max()), 100)
y_line = w0_orig + w1_orig * x_line
plt.plot(x_line, y_line, 'r-', linewidth=2, label='Linear Solution')

plt.xlabel('Floor Size')
plt.ylabel('Price')
plt.title('House Prices: Train, Test, and Linear Fit')
plt.legend()
plt.grid(True)
plt.show()