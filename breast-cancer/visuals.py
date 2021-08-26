import matplotlib.pyplot as pl
import numpy as np
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def get_tree(feature, value, random_state):
    if feature == 'max_depth':
        return DecisionTreeClassifier(max_depth=value, random_state=random_state)
    elif feature == 'min_samples_split':
        return DecisionTreeClassifier(min_samples_split=value, random_state=random_state)
    elif feature == 'min_samples_leaf':
        return DecisionTreeClassifier(min_samples_leaf=value, random_state=random_state)
    
    return DecisionTreeClassifier()


def model_learning_knn(X, y, values, feature='n_neighbors'):
    fig = pl.figure(figsize=(15,10))
    for k, value in enumerate(values):
        knn = KNeighborsClassifier(n_neighbors=value)
        sizes, train_scores, test_scores = learning_curve(knn, X, y, cv=10, n_jobs=-1, verbose=0)

        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)

        # Subplot the learning curve
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = 'g')

        # Labels
        ax.set_title('{} = {}'.format(feature, value))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
    
    # Visual aesthetics
    ax.legend(loc='best')
    fig.suptitle('Decision Tree Regressor Learning Performances: {}'.format(feature), fontsize = 16, y = 1.03)
    fig.tight_layout()
    fig.show()


def model_learning(X, y, feature, values, random_state=1769):
    fig = pl.figure(figsize=(15,10))
    
    for k, value in enumerate(values):
        tree = get_tree(feature=feature, value=value, random_state=random_state)

        sizes, train_scores, test_scores = learning_curve(tree, X, y, cv=10, n_jobs=1, verbose=0)

        # Find the mean and standard deviation for smoothing
        train_std = np.std(train_scores, axis = 1)
        train_mean = np.mean(train_scores, axis = 1)
        test_std = np.std(test_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)

        # Subplot the learning curve
        ax = fig.add_subplot(2, 2, k+1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = 'r')
        ax.fill_between(sizes, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = 'g')

        # Labels
        ax.set_title('{} = {}'.format(feature, value))
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
    
    # Visual aesthetics
    ax.legend(loc='best')
    fig.suptitle('Decision Tree Regressor Learning Performances: {}'.format(feature), fontsize = 16, y = 1.03)
    fig.tight_layout()
    fig.show()


def model_complexity(X, y, feature, values, random_state=1769):
    train_scores, test_scores = validation_curve(DecisionTreeClassifier(random_state=random_state), X, y, param_name=feature, param_range=values, cv=10)

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    pl.figure(figsize=(7, 5))
    pl.title('Decision Tree Regressor Complexity Performance')
    pl.plot(values, train_mean, 'o-', color = 'r', label = 'Training Score')
    pl.plot(values, test_mean, 'o-', color = 'g', label = 'Validation Score')
    pl.fill_between(values, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = 'r')
    pl.fill_between(values, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = 'g')

    # Visual aesthetics
    pl.legend(loc='best')
    pl.xlabel(feature)
    pl.ylabel('Score')
    pl.show()


