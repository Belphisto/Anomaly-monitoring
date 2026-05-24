import numpy as np
import pandas as pd
import math
import os
import csv
import json
import tensorflow as tf
import joblib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import multi_siamese_nn
import utils
import config
import plots
import accuracy


def _save_actual_snippets_num(snn_results_dir, actual_snippets_num):
    snn_params_path = os.path.join(snn_results_dir, 'snn_params.json')
    if not os.path.exists(snn_params_path):
        return

    with open(snn_params_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data['actual_snippets_num'] = int(actual_snippets_num)

    with open(snn_params_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _trim_snippets_to_requested(x_snippets, y_snippets, dataset_params):
    requested_snippets_num = int(dataset_params.get('snippets_number', len(x_snippets)))
    if len(x_snippets) > requested_snippets_num:
        x_snippets = x_snippets[:requested_snippets_num]
        y_snippets = y_snippets[:requested_snippets_num]
    return x_snippets, y_snippets


def _as_1d_numeric_array(x):
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0]
        else:
            x = x.to_numpy().reshape(-1)
    elif isinstance(x, pd.Series):
        x = x.to_numpy()
    else:
        x = np.asarray(x)

    x = np.asarray(x).reshape(-1)

    if x.dtype == object:
        cleaned = []
        for v in x:
            if isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v).reshape(-1)
                if arr.size == 0:
                    continue
                cleaned.extend(arr.tolist())
            else:
                cleaned.append(v)
        x = np.asarray(cleaned)

    return x.astype(float)


def _build_ready_inputs(X, snippets_num=None):
    X = np.asarray(X)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], X.shape[3], 1))

    actual_snippets_num = int(X.shape[1])
    actual_pairs_num = int(X.shape[2])

    if snippets_num is None:
        snippets_num = actual_snippets_num
    else:
        snippets_num = min(int(snippets_num), actual_snippets_num)

    ready_X = []
    for i in range(snippets_num):
        for j in range(actual_pairs_num):
            ready_X.append(X[:, i, j])

    return X, ready_X, snippets_num, actual_pairs_num


def main():

    args = utils.parse_args()

    dataset_dir = os.path.join(config.SNN_DATASETS_DIR, args.dataset, str(args.dimension))
    results_dir = os.path.join(config.RESULTS_DIR, args.dataset, str(args.dimension))
    utils.create_directory(results_dir)

    dataset_params = utils.read_json_file(os.path.join(dataset_dir, 'input_params.json'))

    if args.act == 'fit':

        print('1. Start to read and transform the training set for fitting the SNN')
        train_val_set_path = os.path.join(dataset_dir, 'train_set.csv')
        x_train_val, y_train_val = utils.load_dataset(train_val_set_path)
        x_train_val = utils.normalize_dataset(x_train_val)
        x_train, y_train, x_val, y_val = utils.train_val_split(x_train_val, y_train_val, args.val_size)

        snippets_set_path = os.path.join(dataset_dir, 'snippets.csv')
        x_snippets, y_snippets = utils.load_dataset(snippets_set_path)
        x_snippets = utils.normalize_dataset(x_snippets)
        x_snippets, y_snippets = _trim_snippets_to_requested(x_snippets, y_snippets, dataset_params)

        X_train, Y_train = utils.make_train_set_samples_with_snippets(x_train, y_train, x_snippets, y_snippets)
        X_val, Y_val = utils.make_train_set_samples(x_val, y_val)

        X_train, ready_X_train, actual_snippets_num_train, _ = _build_ready_inputs(X_train)
        X_val, ready_X_val, _, _ = _build_ready_inputs(X_val, snippets_num=actual_snippets_num_train)

        x_snippets = x_snippets[:actual_snippets_num_train]
        y_snippets = y_snippets[:actual_snippets_num_train]

        print('The training and validation sets are ready to fit the SNN\n')

        print('2. Start to fit SNN')
        SNN_results_dir = os.path.join(results_dir, args.nn_type)
        utils.create_directory(SNN_results_dir)

        mDiSSiD_model = multi_siamese_nn.build_mDiSSiD_model(
            snippets_num=actual_snippets_num_train,
            input_shape=(dataset_params['m'], 1),
            base_type=args.nn_type
        )

        mDiSSiD_model.compile(loss=multi_siamese_nn.multi_loss(margin=args.margin), optimizer=args.optimizer)
        mDiSSiD_model.summary()

        mDiSSiD_model.fit(
            ready_X_train, Y_train,
            validation_data=(ready_X_val, Y_val),
            batch_size=args.batch_size,
            epochs=args.epochs,
            shuffle=True,
        )

        model_dir = os.path.join(SNN_results_dir, 'models')
        utils.create_directory(model_dir)
        mDiSSiD_model.save_weights(os.path.join(model_dir, 'weights.weights.h5'))

        print('mDiSSiD model is fitted\n')

        print('3. Start to read and transform the test set for finding the anomaly threshold')
        test_set_path = os.path.join(dataset_dir, 'test_set.csv')
        x_test, y_test = utils.load_dataset(test_set_path)
        x_test = utils.normalize_dataset(x_test)

        X_test, Y_test = utils.make_test_set_samples(x_test, y_test, x_snippets, y_snippets)
        X_test, ready_X_test, _, _ = _build_ready_inputs(X_test, snippets_num=actual_snippets_num_train)

        predictions = mDiSSiD_model.predict(ready_X_test)
        min_predictions = predictions.min(axis=1)

        utils.write_dataset(min_predictions.reshape(-1, 1), SNN_results_dir, 'test_predictions_for_threshold.csv')

        print('The Siamese neural network finished to detect anomalies in the test set\n')

        utils.save_snn_params(
            args.nn_type,
            args.epochs,
            args.batch_size,
            args.margin,
            args.optimizer,
            os.path.join(SNN_results_dir, 'snn_params.json')
        )
        _save_actual_snippets_num(SNN_results_dir, actual_snippets_num_train)

        print('Fitting the SNN and the finding the anomaly threshold are done\n')

    else:

        snn_params = utils.read_json_file(os.path.join(results_dir, args.nn_type + '/snn_params.json'))

        test_original_ts_path = os.path.join(dataset_dir, 'test_original_ts.csv')
        test_ts = utils.load_test_original_ts_from_csv(test_original_ts_path)

        test_label_path = os.path.join(dataset_dir, 'test_label.csv')
        true_label = utils.load_test_original_ts_from_csv(test_label_path)

        test_ts = _as_1d_numeric_array(test_ts)
        true_label = _as_1d_numeric_array(true_label)

        N = len(test_ts) - dataset_params['m'] + 1
        x_test = utils.split_ts_to_subs(test_ts, N, dataset_params['m'])
        x_test = np.asarray(x_test, dtype=float)
        x_test = utils.normalize_dataset(x_test)

        snippets_set_path = os.path.join(dataset_dir, 'snippets.csv')
        x_snippets, y_snippets = utils.load_dataset(snippets_set_path)
        x_snippets = utils.normalize_dataset(x_snippets)

        actual_snippets_num_saved = int(
            snn_params.get('actual_snippets_num', dataset_params.get('snippets_number', len(x_snippets)))
        )
        x_snippets = x_snippets[:actual_snippets_num_saved]
        y_snippets = y_snippets[:actual_snippets_num_saved]

        X_test = utils.make_original_test_ts_samples(x_test, x_snippets)
        X_test, ready_X_test, actual_snippets_num_test, _ = _build_ready_inputs(
            X_test, snippets_num=actual_snippets_num_saved
        )

        SNN_results_dir = os.path.join(results_dir, args.nn_type)

        mDiSSiD_model = multi_siamese_nn.build_mDiSSiD_model(
            snippets_num=actual_snippets_num_test,
            input_shape=(dataset_params['m'], 1),
            base_type=args.nn_type
        )

        model_path = os.path.join(SNN_results_dir, 'models', 'weights.weights.h5')
        if os.path.exists(model_path):
            tf.compat.v1.reset_default_graph()
            mDiSSiD_model.load_weights(model_path)
        else:
            print("The model doesn't exist")

        predictions = mDiSSiD_model.predict(ready_X_test)
        min_predictions = predictions.min(axis=1)

        utils.write_dataset(min_predictions.reshape(-1, 1), SNN_results_dir, 'original_test_predictions.csv')

        print(f"DiSSiD accuracy on dataset: {dataset_params['input_files']}")


if __name__ == '__main__':
    main()
