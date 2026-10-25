import numpy as np

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None  # Model eğitildikten sonra burada saklanacak

    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            theta is d-dimensional numpy vector
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! ** 
        '''
        m = len(y)  # örnek sayısı
        h = self.sigmoid(X @ theta)  # tahmin olasılıkları
        
        # Log-loss
        cost = (1/m) * np.sum(-y * np.log(h) - (1 - y) * np.log(1 - h))
        
        # Regularizasyon (theta_0 hariç)
        reg_theta = theta.copy()
        reg_theta[0] = 0
        reg_cost = (regLambda / (2*m)) * np.sum(reg_theta ** 2)
        
        return cost + reg_cost

    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            theta is d-dimensional numpy vector
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        m = len(y)
        h = self.sigmoid(X @ theta)
        
        # Regularizasyon (theta_0 hariç)
        reg_theta = theta.copy()
        reg_theta[0] = 0
        grad = (1/m) * (X.T @ (h - y)) + (regLambda / m) * reg_theta
        
        return grad
    


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        ** the d here is different from above! (due to augmentation) **
        '''
        n, d = X.shape
        
        # Bias sütunu ekle (1'ler)
        X = np.c_[np.ones((n, 1)), X]
        
        # Theta'yı rastgele başlat (mean 0)
        np.random.seed(42)  # tekrarlanabilirlik için
        self.theta = np.random.randn(d + 1) * 0.01
        
        for i in range(self.maxNumIters):
            theta_old = self.theta.copy()
            
            # Gradyanı hesapla ve güncelle
            grad = self.computeGradient(self.theta, X, y, self.regLambda)
            self.theta = self.theta - self.alpha * grad
            
            # Yakınsama kontrolü
            if np.linalg.norm(self.theta - theta_old) <= self.epsilon:
                print(f"Converged at iteration {i}")
                break


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions, the output should be binary (use h_theta > .5)
        '''
        n = X.shape[0]
        
        # Bias sütunu ekle
        X = np.c_[np.ones((n, 1)), X]
        
        # Olasılıkları hesapla
        h = self.sigmoid(X @ self.theta)
        
        # 0.5'ten büyükse 1, değilse 0
        return (h > 0.5).astype(int)


    def sigmoid(self, Z):
        '''
        Applies the sigmoid function on every element of Z
        Arguments:
            Z can be a (n,) vector or (n , m) matrix
        Returns:
            A vector/matrix, same shape with Z, that has the sigmoid function applied elementwise
        '''
        return 1.0 / (1.0 + np.exp(-Z))
