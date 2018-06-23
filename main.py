from gp import *
from plotters import *
from sklearn.datasets import make_classification
from time import time

random_search=False
# random_search=True

if random_search:
	filePath='random'
	random_search=100000
else:
	filePath='L-BFGS-B'
	random_search=False

# Generate classification by 
data, target = make_classification(n_samples=500,
                                   n_features=45,
                                   n_informative=15,
                                   n_redundant=5)

# The hyperparameter grid space we want to explore
lambdas = np.linspace(1, -4, 25)
gammas = np.linspace(1, -4, 20)

# Find the true loss of the SVM based on the hyperparameters and the data above
print('Eavluating score on the grid space...')
print('This might take a while...')
t1=time()
param_grid = np.array([[C, gamma] for gamma in gammas for C in lambdas])
real_loss = [sample_loss(params, data, target) for params in param_grid]
t=time()-t1
print(t//60, 'minutes', t%60, 'seconds used.')

# The maximum (the best hyperparameter) is at:
opt=param_grid[np.array(real_loss).argmax(), :]
print('Maximum is at [ C, gamma ] =', opt)

# Show real loss for the whole grid space
show_true_loss_func(lambdas, gammas, real_loss, filePath)

# The hyperparameter grid space where we want to run bayesian_optimisation.
# Should be same as the previous grid space.
bounds = np.array([[-4, 1], [-4, 1]])

# Run bayesian optimisation
print('Running bayesian optimisation...')
xp, yp = bayesian_optimisation(n_iters=30, 
                               sample_loss=sample_loss, 
                               bounds=bounds,
                               n_pre_samples=3,
                               random_search=random_search,
                               data=data,
                               target=target)

# Show bayesian optimisation process.
from matplotlib import rc
rc('text', usetex=False)
plot_iteration(filePath, lambdas, xp, yp, first_iter=3, second_param_grid=gammas, optimum=opt)