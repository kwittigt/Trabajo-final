import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializa los pesos y sesgos de la capa oculta y de salida
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # Pesos de entrada a oculta
        self.b1 = np.zeros((1, hidden_size))                       # Sesgo de la capa oculta
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01 # Pesos de oculta a salida
        self.b2 = np.zeros((1, output_size))                       # Sesgo de la capa de salida
    
    def relu(self, x):
        # Función de activación ReLU: reemplaza negativos por 0
        return np.maximum(0, x)
    
    def softmax(self, x):
        # Función softmax: convierte los valores en probabilidades
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Estabilización numérica
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # Propagación hacia adelante
        self.z1 = np.dot(X, self.W1) + self.b1      # Entrada a la capa oculta
        self.a1 = self.relu(self.z1)                # Salida de la capa oculta (ReLU)
        self.z2 = np.dot(self.a1, self.W2) + self.b2# Entrada a la capa de salida
        self.a2 = self.softmax(self.z2)             # Salida de la capa de salida (Softmax)
        return self.a2
    
    def compute_loss(self, y, y_pred):
        # Calcula la pérdida de entropía cruzada
        m = y.shape[0]
        log_probs = -np.log(y_pred[range(m), y.argmax(axis=1)])
        return np.sum(log_probs) / m
    
    def backward(self, X, y, y_pred, learning_rate):
        m = X.shape[0]
        
        # Gradiente para la capa de salida
        dz2 = y_pred - y
        dw2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Gradiente para la capa oculta
        dz1 = np.dot(dz2, self.W2.T) * (self.z1 > 0)  # Derivada de ReLU
        dw1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Actualiza los pesos y sesgos usando descenso de gradiente
        self.W2 -= learning_rate * dw2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dw1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, learning_rate=0.01, epochs=1000):
        # Entrena la red neuronal durante varias épocas
        loss_history = []
        for epoch in range(epochs):
            y_pred = self.forward(X)                # Propagación hacia adelante
            loss = self.compute_loss(y, y_pred)     # Calcula la pérdida
            self.backward(X, y, y_pred, learning_rate) # Propagación hacia atrás y actualización
            loss_history.append(loss)               # Guarda la pérdida de cada época
        return loss_history
    
    def predict(self, X):
        # Predice la clase para cada muestra de entrada
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1) + 1  # Devuelve clases 1, 2, 3
    
    def accuracy(self, X, y):
        # Calcula la precisión del modelo
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1) + 1
        return np.mean(predictions == true_labels)