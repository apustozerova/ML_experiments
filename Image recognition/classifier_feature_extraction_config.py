# Values for parameter selection for testing classification algorithms
# MLP: size, activation and solver
mlp_size = [100, 120, 150, 200, 250, 300, 500, 700, 1000, 1250, 1500, 3000, 5000] 
mlp_activation = ['identity', 'logistic', 'tanh', 'relu']
mlp_solver = ['lbfgs', 'sgd', 'adam']

# Random Forest: size
rf_size = [3, 5, 7, 10, 15, 20, 25, 30, 35, 50]

# Knn: number of neighbors
knn_k = [3, 5, 7, 10, 20, 30]
