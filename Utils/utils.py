import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from skorch.callbacks import EpochScoring
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from torch import nn
from skorch import NeuralNetClassifier

from Utils.NeuralNetworks import SigmoidNeuralNetwork

def load_data(seed: int, feature_size: int, discreteize: bool = False, expand: bool = False, expansion_degree: int = 6):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    X = np.load('../Datasets/kryptonite-%s-X.npy'%(feature_size))
    y = np.load('../Datasets/kryptonite-%s-y.npy'%(feature_size))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    scaler = StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if discreteize:
        X_train = np.where(X_train >= 0, np.float32(1.0), np.float32(-1.0))
        X_test = np.where(X_test >= 0, np.float32(1.0), np.float32(-1.0))

    if expand:
        X_train = PolynomialFeatures(degree=expansion_degree).fit_transform(X_train)
        X_test = PolynomialFeatures(degree=expansion_degree).fit_transform(X_test)

    return X_train, np.int32(y_train), X_test, np.int32(y_test), X_test.shape[1]

def run_multiple_nn(
        gs,
        base_feature_size: int,
        device: str,
        no_runs: int
        ):
    
    results = []

    for run_index in range(no_runs):
        print(f'Run {run_index + 1} / {no_runs} started')

        torch.manual_seed(run_index)
        torch.cuda.manual_seed(run_index)
        np.random.seed(run_index)

        X_train, y_train, X_test, y_test, feature_size = load_data(run_index, base_feature_size)

        net = NeuralNetClassifier(
            module=SigmoidNeuralNetwork,
            max_epochs=100,
            criterion=nn.BCEWithLogitsLoss,
            optimizer=torch.optim.Adam,
            optimizer__lr=gs.best_params_['optimizer__lr'],
            optimizer__weight_decay=gs.best_params_['optimizer__weight_decay'],
            batch_size=gs.best_params_['batch_size'],
            module__input_size=feature_size,
            module__layer_size=gs.best_params_['module__layer_size'],
            device=device,
            callbacks=[
                EpochScoring(scoring='accuracy', name='train_acc', on_train=True),
            ],
            train_split=None,
            verbose=0
        )

        net.fit(X_train, y_train.astype(np.float32))

        train_loss = net.history[:, 'train_loss']
        train_acc = net.history[:, 'train_acc']

        test_accuracy = net.score(X_test, y_test)

        for epoch, loss, acc in zip(np.arange(100), train_loss, train_acc):
            results.append({
                'Seed': run_index,
                'epoch': epoch,
                'train_loss': loss,
                'train_accuracy': acc,
                'test_accuracy': test_accuracy
            })

        print(f'Run finished with test accuracy: {test_accuracy}')

    return pd.DataFrame(results)

def run_multiple_forest(
        base_feature_size: int,
        no_runs: int
        ):
    
    results = []

    for run_index in range(no_runs):
        print("Run %d / %d started" % (run_index + 1, no_runs))

        X_train, y_train, X_test, y_test, _  = load_data(run_index, base_feature_size, discreteize=True, expand=True)

        forrest = RandomForestClassifier(min_samples_leaf=10, n_jobs=-1).fit(X_train, y_train)

        test_accuracy = forrest.score(X_test, y_test)

        results.append({
            'run': run_index,
            'test_accuracy': test_accuracy
        })

        print(f'Run finished with test accuracy: {test_accuracy}')

    return pd.DataFrame(results)

def run_multiple_logistics(
        gs,
        base_feature_size: int,
        no_runs: int
        ):
    
    results = []

    for run_index in range(no_runs):
        print("Run %d / %d started" % (run_index + 1, no_runs))
        
        X_train, y_train, X_test, y_test, _  = load_data(run_index, base_feature_size, discreteize=True, expand=True)

        logreg = SGDClassifier(
            loss='log_loss',
            alpha=gs.best_params_['alpha'],
            max_iter=gs.best_params_['max_iter'],
            tol=gs.best_params_['tol'],
            learning_rate=gs.best_params_['learning_rate'],
            eta0=gs.best_params_['eta0'],
            n_jobs=-1).fit(X_train, y_train)

        test_accuracy = logreg.score(X_test, y_test)

        results.append({
            'run': run_index,
            'test_accuracy': test_accuracy
        })

        print(f'Run finished with test accuracy: {test_accuracy}')

    return pd.DataFrame(results)