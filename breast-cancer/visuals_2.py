import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV


def render_confusion_matrix(y_test, y_pred, ratio=False):
    print(classification_report(y_test,y_pred))
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=['Positive', 'Negative'], columns=['Positive', 'Negative'])
    if ratio:
        cm_df = (cm_df/(cm_df.values.sum())*100).round(2)

    sns.heatmap(cm_df, annot=True)


def plot_learning_curve(estimator, X, y, title='Learning Curve', axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), verbose=0):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def plot_learning_curve_all(estimator, X, y, param, values, title='Learning Curve', axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), verbose=0):
    fig, axes = plt.subplots(3, len(values), figsize=(15,10))
    for indx, value in enumerate(values):
        clonsed_estimator = clone(estimator)
        estimator.set_params(**{param: value})
        plot_learning_curve(estimator, X, y, title=f'{param}: {value}', axes=axes[:, indx], ylim=ylim, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    fig.tight_layout(pad=3.0)
    plt.show()


def model_complexity(estimator, X, y, feature, values, cv=None, verbose=0, n_jobs=None):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=feature, param_range=values, cv=cv, verbose=verbose, n_jobs=n_jobs)

    # Find the mean and standard deviation for smoothing
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot the validation curve
    plt.figure(figsize=(7, 5))
    plt.grid()
    plt.title('Classifier Complexity Performance')
    plt.plot(values, train_mean, 'o-', color = 'r', label = 'Training Score')
    plt.plot(values, test_mean, 'o-', color = 'g', label = 'Validation Score')
    plt.fill_between(values, train_mean - train_std, train_mean + train_std, alpha = 0.15, color = 'r')
    plt.fill_between(values, test_mean - test_std, test_mean + test_std, alpha = 0.15, color = 'g')

    # Visual aesthetics
    plt.legend(loc='best')
    plt.xlabel(feature)
    plt.ylabel('Score')
    plt.show()


def grid_search_and_plot(estimator, X, y, param, cv=None, verbose=0, n_jobs=None, plot=True):
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param,
        cv=cv,
        verbose=verbose,
        n_jobs=n_jobs
    )
    grid.fit(X, y)

    if plot:
        for k,v in param.items():
            model_complexity(estimator, X, y, feature=k, values=v, cv=cv, verbose=verbose, n_jobs=n_jobs)
    
    print('Best score from the Grid Search:', grid.best_score_)
    return grid

            

