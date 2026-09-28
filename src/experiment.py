"""
Choose a hidden-layer size with time-series validation, then test each model.
"""
import numpy as np
import pandas as pd

from datasets import ROOT, DATA_SPECS, load_data, transform, windows, inverse
from rnn import RNN, train

MODELS = ['Elman', 'Jordan', 'Multi']
HIDDEN_SIZES = [4, 8, 16]
TEST_SEEDS = [1, 2, 3]
WINDOW = 12
MAX_EPOCHS = 200


def errors(actual, predicted):
    difference = actual - predicted
    return {'rmse': np.sqrt(np.mean(difference ** 2)),
            'mae': np.mean(np.abs(difference))}


def prepare_split(transformed, offset, observed, train_end, test_end):
    """Fit the scaler on the training prefix, then make past-only windows."""
    mean = transformed[:train_end - offset].mean()
    scale = transformed[:train_end - offset].std()
    train_indices = np.arange(offset + WINDOW, train_end)
    test_indices = np.arange(train_end, test_end)
    train_indices = train_indices[observed[train_indices]]
    test_indices = test_indices[observed[test_indices]]
    x_train, y_train = windows(transformed, train_indices, offset, mean, scale, WINDOW)
    x_test, y_test = windows(transformed, test_indices, offset, mean, scale, WINDOW)
    return x_train, y_train, x_test, y_test, test_indices, mean, scale


def main():
    cv_rows, selected_rows, test_rows, prediction_rows, history_rows = [], [], [], [], []

    for name, frame in load_data().items():
        settings = DATA_SPECS[name]
        values = frame.value.to_numpy()
        observed = np.isfinite(values)
        raw, transformed, offset = transform(values, settings['transform'])
        count = len(raw)
        test_start = int(0.8 * count)
        folds = [(int(0.5 * count), int(0.6 * count)),
                 (int(0.6 * count), int(0.7 * count)),
                 (int(0.7 * count), test_start)]

        for kind in MODELS:
            best_score = float('inf')
            for hidden_size in HIDDEN_SIZES:
                fold_scores, stopping_epochs = [], []
                for fold, (train_end, validation_end) in enumerate(folds, start=1):
                    x, y, xv, yv, indices, mean, scale = prepare_split(
                        transformed, offset, observed, train_end, validation_end)
                    model = RNN(kind, hidden_size, seed=0)
                    epoch, history = train(model, x, y, MAX_EPOCHS, validation=(xv, yv))
                    predicted = inverse(model.predict(xv) * scale + mean,
                                        indices, raw, settings['transform'])
                    score = errors(raw[indices], predicted)
                    fold_scores.append(score['rmse'])
                    stopping_epochs.append(epoch)
                    cv_rows.append(dict(dataset=name, model=kind, hidden=hidden_size,
                                        fold=fold, epochs=epoch, **score))

                # Choose the width by mean validation RMSE. Use the median stopping
                # epoch  for the final fit.
                if np.mean(fold_scores) < best_score:
                    best_score = np.mean(fold_scores)
                    best_hidden = hidden_size
                    best_epochs = int(np.median(stopping_epochs))
                    best_history = history  # Most recent validation fold.

            selected_rows.append(dict(dataset=name, model=kind, hidden=best_hidden,
                                      epochs=best_epochs, cv_rmse=best_score))
            for epoch, training_loss, validation_loss in best_history:
                history_rows.append(dict(dataset=name, model=kind, epoch=epoch,
                                         train_mse=training_loss, validation_mse=validation_loss))

            # Refit on the first 80%. The final 20% never selects any setting.
            x, y, xt, _, indices, mean, scale = prepare_split(
                transformed, offset, observed, test_start, count)
            for seed in TEST_SEEDS:
                model = RNN(kind, best_hidden, seed)
                train(model, x, y, epochs=best_epochs)
                predicted = inverse(model.predict(xt) * scale + mean,
                                    indices, raw, settings['transform'])
                test_rows.append(dict(dataset=name, model=kind, seed=seed,
                                      **errors(raw[indices], predicted)))
                for index, prediction in zip(indices, predicted):
                    prediction_rows.append(dict(dataset=name, model=kind, seed=seed,
                                                date=frame.date.iloc[index], actual=raw[index],
                                                prediction=prediction))
            print(f'{name}: {kind}, hidden={best_hidden}, epochs={best_epochs}', flush=True)

        # Simple references: repeat the last value, add a training-period drift
        # or repeat the same month last year for the monthly series.
        baselines = {'Naive': raw[indices - 1],
                     'Drift': raw[indices - 1] + np.diff(raw[:test_start]).mean()}
        if settings['period'] == 12:
            baselines['Seasonal naive'] = raw[indices - 12]
        for kind, predicted in baselines.items():
            test_rows.append(dict(dataset=name, model=kind, seed=0,
                                  **errors(raw[indices], predicted)))

    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    pd.DataFrame(cv_rows).to_csv(output / 'cv.csv', index=False)
    pd.DataFrame(selected_rows).to_csv(output / 'selected.csv', index=False)
    pd.DataFrame(test_rows).to_csv(output / 'runs.csv', index=False)
    pd.DataFrame(prediction_rows).to_csv(output / 'predictions.csv', index=False)
    pd.DataFrame(history_rows).to_csv(output / 'histories.csv', index=False)
    summary = pd.DataFrame(test_rows).groupby(['dataset', 'model']).agg(
        rmse=('rmse', 'mean'), rmse_sd=('rmse', 'std'), mae=('mae', 'mean'))
    summary.to_csv(output / 'summary.csv')
    print(summary.round(3))


if __name__ == '__main__':
    main()
