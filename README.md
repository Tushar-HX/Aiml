#lab1
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (example dataset from URL)
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)

# Display dataset info
print(df.head())

# Split dataset into features (X) and target (y)
X = df[['Hours']]
y = df['Scores']

# Split into training and testing sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Visualization
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel("Study Hours")
plt.ylabel("Scores")
plt.title("Linear Regression - Student Marks Prediction")
plt.legend()
plt.show()

#lab1pt.2
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

# Load breast cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
clf = LogisticRegression(max_iter=2000)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation Metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues",
            xticklabels=cancer.target_names,
            yticklabels=cancer.target_names,
            fmt="d")

plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix - Breast Cancer Dataset")
plt.tight_layout()
plt.show()





#lab02
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load datasets
iris = load_iris()
wine = load_wine()
cancer = load_breast_cancer()

X1 = iris.data[:, :2]
X2 = wine.data[:, :2]
X3 = cancer.data[:, :2]

# Standardization
scaler = StandardScaler()
X1 = scaler.fit_transform(X1)
X2 = scaler.fit_transform(X2)
X3 = scaler.fit_transform(X3)

# KMeans Models
kmeans1 = KMeans(n_clusters=3, random_state=42).fit(X1)
kmeans2 = KMeans(n_clusters=3, random_state=42).fit(X2)
kmeans3 = KMeans(n_clusters=3, random_state=42).fit(X3)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X1[:, 0], X1[:, 1], c=kmeans1.labels_, cmap='cool', label='Iris Dataset')
plt.scatter(X2[:, 0], X2[:, 1], c=kmeans2.labels_, cmap='rainbow', label='Wine Dataset')
plt.scatter(X3[:, 0], X3[:, 1], c=kmeans3.labels_, cmap='winter', label='Breast Cancer Dataset')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering on 3 Inbuilt Datasets")
plt.legend()
plt.grid(True)
plt.show()
#lab2pt2
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load datasets
iris = load_iris()
wine = load_wine()
cancer = load_breast_cancer()

X1 = iris.data[:, :2]
X2 = wine.data[:, :2]
X3 = cancer.data[:, :2]

# Standardization
scaler = StandardScaler()
X1 = scaler.fit_transform(X1)
X2 = scaler.fit_transform(X2)
X3 = scaler.fit_transform(X3)

# KMeans Models
kmeans1 = KMeans(n_clusters=3, random_state=42).fit(X1)
kmeans2 = KMeans(n_clusters=3, random_state=42).fit(X2)
kmeans3 = KMeans(n_clusters=3, random_state=42).fit(X3)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X1[:, 0], X1[:, 1], c=kmeans1.labels_, cmap='cool', label='Iris Dataset')
plt.scatter(X2[:, 0], X2[:, 1], c=kmeans2.labels_, cmap='rainbow', label='Wine Dataset')
plt.scatter(X3[:, 0], X3[:, 1], c=kmeans3.labels_, cmap='winter', label='Breast Cancer Dataset')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering on 3 Inbuilt Datasets")
plt.legend()
plt.grid(True)
plt.show()






#lab03
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load datasets
datasets = [load_iris(), load_wine(), load_breast_cancer()]
labels = ['Iris', 'Wine', 'Breast Cancer']

plt.figure(figsize=(10, 6))

# Compression factor (reduces spread to make clusters tight)
compression_factor = 0.5

for data, name in zip(datasets, labels):
    X = data.data
    X = StandardScaler().fit_transform(X)

    # PCA to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Apply compression
    X_compressed = X_pca * compression_factor

    # KMeans
    km = KMeans(n_clusters=3, random_state=42)
    clusters = km.fit_predict(X_compressed)

    plt.scatter(X_compressed[:, 0], X_compressed[:, 1], label=name)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA + K-Means (Compressed / Tight Clusters)")
plt.legend()
plt.grid(True)
plt.show()
#lab03pt2
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data                     # 4 features
y = iris.target.reshape(-1, 1)    # target labels

# One-hot encode target labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Initialize network parameters
np.random.seed(42)
input_size = X_train.shape[1]     # 4 features
hidden_size = 6                   # 6 neurons in hidden layer
output_size = y_train.shape[1]    # 3 classes
learning_rate = 0.01
epochs = 500

# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    n = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / n

# Step 3: Training (Forward + Backward Propagation)
losses = []

for epoch in range(epochs):
    # ---- Forward Propagation ----
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = softmax(z2)

    # ---- Compute Loss ----
    loss = cross_entropy(y_train, y_pred)
    losses.append(loss)

    # ---- Backward Propagation ----
    m = X_train.shape[0]

    # Output layer error
    dz2 = (y_pred - y_train) / m
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    # Hidden layer error
    dz1 = np.dot(dz2, W2.T) * a1 * (1 - a1)
    dW1 = np.dot(X_train.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # ---- Update Weights ----
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

# Step 4: Evaluate model accuracy
def predict(X):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = softmax(z2)
    return np.argmax(y_pred, axis=1)

y_pred_test = predict(X_test)
y_true_test = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred_test == y_true_test) * 100
print(f"\nModel Accuracy: {accuracy:.2f}%")

# Step 5: Plot loss curve
plt.figure(figsize=(8,5))
plt.plot(losses, label="Cross-Entropy Loss", color='blue')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve of ANN on Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()









#lab04
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# --- Part 1: Visualization of Functions and Gradients ---

print("--- 1. Generating Activation Function Plots ---")

# Define the input range
x = np.linspace(-10, 10, 100)

# Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Define derivatives
def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def relu_prime(x):
    return np.where(x > 0, 1, 0)

# Create plots
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot 1: Activation Functions
ax1.plot(x, sigmoid(x), label='Sigmoid', c='blue')
ax1.plot(x, tanh(x), label='Tanh', c='green')
ax1.plot(x, relu(x), label='ReLU', c='red')
ax1.set_title('Activation Functions')
ax1.set_ylabel('Output')
ax1.legend()
ax1.axhline(0, color='black', linewidth=0.5)
ax1.axvline(0, color='black', linewidth=0.5)

# Plot 2: Gradients
ax2.plot(x, sigmoid_prime(x), label="Sigmoid Derivative", c='blue', linestyle='--')
ax2.plot(x, tanh_prime(x), label="Tanh Derivative", c='green', linestyle='--')
ax2.plot(x, relu_prime(x), label="ReLU Derivative", c='red', linestyle='--')
ax2.set_title('Gradients (Derivatives)')
ax2.set_xlabel('Input (x)')
ax2.set_ylabel('Gradient Value')
ax2.legend()
ax2.set_ylim(-0.2, 1.2) # Set Y-limit for better gradient visualization

plt.tight_layout()
plt.show()

print("--- 2. Starting MNIST Model Training ---")

# --- Part 2: Implementation on MNIST Dataset ---

# 1. Load and preprocess the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 2. Define the model-building function
def build_model(activation_func):
    model = Sequential([
        Dense(128, activation=activation_func, input_shape=(784,)),
        Dense(64, activation=activation_func),
        Dense(10, activation='softmax') # Output layer as requested
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Train models for each activation
activations = ['relu', 'sigmoid', 'tanh']
history_data = {}
num_epochs = 10

for activation in activations:
    print(f"\n--- Training model with {activation.upper()} for {num_epochs} epochs ---")
    model = build_model(activation)
    history = model.fit(x_train, y_train,
                        epochs=num_epochs,
                        batch_size=128,
                        validation_data=(x_test, y_test),
                        verbose=1)
    history_data[activation] = history

# 4. Plot and analyze results
print("\n--- 3. Generating Model Performance Plots ---")
plt.figure(figsize=(12, 5))

# Plot Validation Accuracy
plt.subplot(1, 2, 1)
for activation, history in history_data.items():
    plt.plot(history.history['val_accuracy'], label=f'{activation} val_accuracy')
plt.title('Model Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Validation Loss
plt.subplot(1, 2, 2)
for activation, history in history_data.items():
    plt.plot(history.history['val_loss'], label=f'{activation} val_loss')
plt.title('Model Validation Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()








#lab5n6
import random
import time
import matplotlib.pyplot as plt

# -------------------------
# Environment Class
# -------------------------
class Environment:
    def init(self):
        self.rooms = {
            "A": random.choice(["Clean", "Dirty"]),
            "B": random.choice(["Clean", "Dirty"]),
            "C": random.choice(["Clean", "Dirty"]),
            "D": random.choice(["Clean", "Dirty"]),
        }
        self.agent_location = random.choice(list(self.rooms.keys()))
        self.score = 0

    def display(self):
        print("\nEnvironment State:")
        print(f"A: {self.rooms['A']} | B: {self.rooms['B']}")
        print(f"C: {self.rooms['C']} | D: {self.rooms['D']}")
        print(f"Agent is at: {self.agent_location}")
        print(f"Score: {self.score}")
        print("-" * 40)

    def regenerate_dirt(self, probability=0.2):
        for room in self.rooms:
            if self.rooms[room] == "Clean" and random.random() < probability:
                self.rooms[room] = "Dirty"
                print(f"Room {room} became dirty again")


# -------------------------
# Vacuum Agent Class
# -------------------------
class VacuumAgent:
    def init(self, environment):
        self.env = environment
        self.performance_log = []

    def perceive_and_act(self):
        location = self.env.agent_location
        status = self.env.rooms[location]

        if status == "Dirty":
            print(f"Cleaning room {location}")
            self.env.rooms[location] = "Clean"
            self.env.score += 10
        else:
            # Move to a random different room
            next_room = random.choice(list(self.env.rooms.keys()))
            while next_room == location:
                next_room = random.choice(list(self.env.rooms.keys()))

            print(f"Moving from {location} to {next_room}")
            self.env.agent_location = next_room
            self.env.score -= 1

        self.performance_log.append(self.env.score)


# -------------------------
# Simulation Function
# -------------------------
def run_simulation(steps=15):
    env = Environment()
    agent = VacuumAgent(env)

    print("Initial Environment:")
    env.display()

    for step in range(steps):
        print(f"\nStep {step + 1}/{steps}")
        agent.perceive_and_act()
        env.regenerate_dirt()
        env.display()
        time.sleep(0.5)

    print("\nSimulation Complete")
    print(f"Final Room States: {env.rooms}")
    print(f"Final Score: {env.score}")

    # Plot Performance
    plt.figure(figsize=(8, 4))
    plt.plot(agent.performance_log, marker='o')
    plt.title("Performance Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Performance Score")
    plt.grid(True)
    plt.show()


# -------------------------
# Main Execution
# -------------------------
if name == "main":
    run_simulation(steps=20)









#lab07
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Step 1: Define Rastrigin Function
# ------------------------------
def rastrigin(X):
    x, y = X
    return 20 + x*2 + y2 - 10(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

# ------------------------------
# Step 2: Initialize Population
# ------------------------------
def initialize_population(pop_size, bounds):
    # Randomly initialize individuals (x, y) within given bounds
    population = np.random.uniform(low=bounds[0], high=bounds[1], size=(pop_size, 2))
    return population

# ------------------------------
# Step 3: Fitness Function
# ------------------------------
def fitness(population):
    # Inverse of Rastrigin value (since we minimize)
    return 1 / (1 + np.array([rastrigin(ind) for ind in population]))

# ------------------------------
# Step 4: Selection (Roulette Wheel)
# ------------------------------
def select_parents(population, fitness_vals):
    probs = fitness_vals / np.sum(fitness_vals)
    selected_idx = np.random.choice(len(population), size=2, replace=False, p=probs)
    return population[selected_idx]

# ------------------------------
# Step 5: Crossover (Blending)
# ------------------------------
def crossover(parent1, parent2):
    alpha = np.random.rand()  # Random blending coefficient
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = alpha * parent2 + (1 - alpha) * parent1
    return child1, child2

# ------------------------------
# Step 6: Mutation
# ------------------------------
def mutate(child, mutation_rate=0.1):
    if np.random.rand() < mutation_rate:
        mutation = np.random.uniform(-0.5, 0.5, size=2)
        child += mutation
    return child

# ------------------------------
# Step 7: Plot Heatmap Function
# ------------------------------
def plot_heatmap(bounds, population=None, best=None, generation=None):
    # Create grid for visualization
    x = np.linspace(bounds[0], bounds[1], 200)
    y = np.linspace(bounds[0], bounds[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = 20 + X*2 + Y2 - 10(np.cos(2*np.pi*X) + np.cos(2*np.pi*Y))

    plt.figure(figsize=(7, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='plasma')
    plt.colorbar(label='Rastrigin Function Value')

    # Plot population points
    if population is not None:
        plt.scatter(population[:, 0], population[:, 1], color='white', edgecolors='black', label='Population')

    # Mark best solution
    if best is not None:
        plt.scatter(best[0], best[1], color='red', s=80, marker='*', label='Best Solution')

    plt.title(f"GA Population - Generation {generation}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.show()

# ------------------------------
# Step 8: Main GA Function
# ------------------------------
def genetic_algorithm(generations=50, pop_size=30, bounds=[-5.12, 5.12]):
    population = initialize_population(pop_size, bounds)
    best_scores = []
    print("Starting Genetic Algorithm...\n")

    for gen in range(generations):
        fitness_vals = fitness(population)
        new_population = []

        # Keep best individual (elitism)
        best_idx = np.argmax(fitness_vals)
        best_individual = population[best_idx]
        best_scores.append(rastrigin(best_individual))

        # Print progress
        if gen % 10 == 0 or gen == generations - 1:
            print(f"Generation {gen}: Best Fitness Value = {rastrigin(best_individual):.4f}")
            # Visualize population movement on heatmap
            plot_heatmap(bounds, population, best_individual, generation=gen)

        # Create new generation
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, fitness_vals)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            new_population.extend([child1, child2])

        population = np.array(new_population[:pop_size])

    # Final best individual
    best_idx = np.argmax(fitness(population))
    best_solution = population[best_idx]
    print("\nBest Solution Found:")
    print(f"x = {best_solution[0]:.4f}, y = {best_solution[1]:.4f}")
    print(f"Rastrigin Value = {rastrigin(best_solution):.4f}")

    # Plot convergence
    plt.figure(figsize=(7,5))
    plt.plot(best_scores, color='purple', linewidth=2)
    plt.title("Genetic Algorithm Convergence")
    plt.xlabel("Generations")
    plt.ylabel("Best Rastrigin Value")
    plt.grid(True)
    plt.show()

# ------------------------------
# Step 9: Run GA
# ------------------------------
genetic_algorithm()












#lab08
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from mpl_toolkits.mplot3d import Axes3D

# -------------------------
# Step 1: Define coefficients
# -------------------------
# Objective: Maximize Z = 30x1 + 20x2 (linprog minimizes, so use negative)
c = [-30, -20]

# -------------------------
# Step 2: Define constraints
# -------------------------
# 2x1 + 3x2 ≤ 100 (Labor)
# 3x1 + 2x2 ≤ 90  (Material)
A = [
    [2, 3],
    [3, 2]
]
b = [100, 90]

# -------------------------
# Step 3: Define variable bounds
# -------------------------
x_bounds = (0, None)
bounds = [x_bounds, x_bounds]

# -------------------------
# Step 4: Solve optimization problem
# -------------------------
result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

if result.success:
    x1, x2 = result.x
    print("✅ Optimization Successful!")
    print(f"Optimal Production of Product A: {x1:.2f} units")
    print(f"Optimal Production of Product B: {x2:.2f} units")
    print(f"Maximum Profit: ₹{(-result.fun):.2f}")
else:
    print("❌ Optimization Failed:", result.message)

# -------------------------
# Step 5: 2D Feasible Region Visualization
# -------------------------
x = np.linspace(0, 50, 200)
y1 = (100 - 2*x) / 3   # Labor constraint
y2 = (90 - 3*x) / 2    # Material constraint

plt.figure(figsize=(8,6))
y3 = np.minimum(y1, y2)
y3[y3 < 0] = np.nan

plt.fill_between(x, y3, color='lightgreen', alpha=0.4, label="Feasible Region")
plt.plot(x, y1, color='red', label=r'2x₁ + 3x₂ ≤ 100')
plt.plot(x, y2, color='blue', label=r'3x₁ + 2x₂ ≤ 90')
plt.scatter(x1, x2, color='black', s=200, marker='*', label='Optimal Point')
plt.xlabel("Product A (x₁)")
plt.ylabel("Product B (x₂)")
plt.title("2D Feasible Region for Resource Allocation")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Step 6: 3D Profit Surface Visualization
# -------------------------
# Create meshgrid for visualization
X, Y = np.meshgrid(np.linspace(0, 50, 100), np.linspace(0, 50, 100))
Z = 30*X + 20*Y  # Profit function

# Apply constraints (make infeasible points NaN)
Z[(2*X + 3*Y > 100) | (3*X + 2*Y > 90)] = np.nan

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9, edgecolor='none')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Profit (₹)")

# Mark feasible region boundary
ax.plot(x, y1, 0, color='red', linewidth=2, label=r'Labor Constraint')
ax.plot(x, y2, 0, color='blue', linewidth=2, label=r'Material Constraint')

# Mark optimal point in 3D
ax.scatter(x1, x2, 30*x1 + 20*x2, color='black', s=100, marker='*', label='Optimal Solution')

# Labels
ax.set_xlabel("Product A (x₁)")
ax.set_ylabel("Product B (x₂)")
ax.set_zlabel("Profit (₹)")
ax.set_title("3D Profit Surface with Constraints")
ax.legend()
plt.show()
