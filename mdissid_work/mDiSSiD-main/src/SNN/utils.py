import numpy as np
import pandas as pd
import random
import csv
import os
import json
import argparse
import joblib
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import config


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', '--dataset_name', action='store', dest='dataset', help='Dataset name', required=True)
    parser.add_argument('-dimension', '--dimension', action='store', dest='dimension', type=int, help='The number of dimension', required=True)

    parser.add_argument('-nn_type', '--nn_type', action='store', dest='nn_type', default=config.NN_TYPE, help='Neural Network type')
    parser.add_argument('-act', '--act', action='store', dest='act', choices=['fit', 'detect'], default=config.ACT, help='Choose fit the neural network or detect anomalies in the test set')

    parser.add_argument('-epochs', '--epochs', action='store', dest='epochs', default=config.EPOCHS, type=int, help='Epoch count')
    parser.add_argument('-batch_size', '--batch_size', action='store', dest='batch_size', default=config.BATCH_SIZE, type=int, help='Batch size')
    parser.add_argument('-margin', '--margin', action='store', dest='margin', default=config.MARGIN, type=float, help='Margin')
    parser.add_argument('-l_emb', '--l_emb', action='store', dest='l_emb', default=config.L_EMB, type=int, help='Subsequence length for MPdist')
    parser.add_argument('-mpdist_k', '--mpdist_k', action='store', dest='mpdist_k', default=config.MPDIST_K, type=float, help='Top-k distance in P_ABBA for MPdist')
    parser.add_argument('-optimizer', '--optimizer', action='store', dest='optimizer', default=config.OPTIMIZER, help='Optimizer')
    parser.add_argument('-val_size', '--val_size', action='store', dest='val_size', default=config.VAL_SIZE, type=float, help='Fraction of validation dataset')

    return parser.parse_args()


def parse_predict_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', '--dataset_name', action='store', dest='dataset', help='Dataset name', required=True)
    parser.add_argument('-nn_type', '--nn_type', action='store', dest='nn_type', default=config.NN_TYPE, help='Neural Network type')

    return parser.parse_args()


def create_directory(directory_path):

    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except Exception:
            return None
        return directory_path


def read_json_file(file_path):

    with open(file_path) as json_file:
        data = json.load(json_file)

    return data


def load_dataset(file_path):
    dataset = pd.read_csv(file_path, header=None)

    X = dataset.iloc[:, 0:-1]
    y = dataset.iloc[:, -1]

    return X, y


def load_test_original_ts_from_txt(file_path) -> np.array:
    input_file = open(file_path, "r")

    ts = input_file.read().split("\n")
    ts = np.array([float(elem) for elem in ts if elem != ""])

    return ts


def load_test_original_ts_from_csv(file_path):
    data = pd.read_csv(file_path, index_col=None, header=None)
    return data.values


def split_ts_to_subs(ts, subs_count, m):

    subs = []

    for i in range(subs_count):
        subs.append(list(ts[i:i + m]))

    return pd.DataFrame(subs)


def _to_numeric_ndarray(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy()
    elif isinstance(X, pd.Series):
        arr = X.to_numpy()
    else:
        arr = np.asarray(X)

    if arr.dtype == object:
        rows = []
        for item in arr:
            if isinstance(item, (list, tuple, np.ndarray)):
                rows.append(np.asarray(item, dtype=float).reshape(-1))
            else:
                rows.append(np.asarray([item], dtype=float))
        try:
            arr = np.vstack(rows)
        except Exception:
            arr = np.asarray(rows, dtype=float)

    arr = np.asarray(arr, dtype=float)
    return arr


def normalize_dataset(X) -> np.array:
    X = _to_numeric_ndarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X.T)
    X_norm = scaler.transform(X.T).T

    return X_norm


def train_val_split(X, Y, val_size=0.2) -> (np.array, np.array, np.array, np.array):
    df = pd.DataFrame(X)
    X = df.values.tolist()
    X = np.array(X)

    df = pd.DataFrame(Y)
    Y = df.values.tolist()
    y = np.array([y_i[0] for y_i in Y])

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=42)

    return x_train, y_train, x_val, y_val


def make_train_set_pairs(x, y) -> (np.array, np.array):
    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []
    num_pos_pairs_per_subs = 5
    num_neg_pairs_per_subs = 5

    for idx1 in range(len(x)):
        x1 = x[idx1]
        label1 = y[idx1]

        for _ in range(num_pos_pairs_per_subs):
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]

            pairs += [[x1, x2]]
            labels += [0]

        for _ in range(num_neg_pairs_per_subs):
            label2 = random.randint(0, num_classes - 1)
            while label2 == label1:
                label2 = random.randint(0, num_classes - 1)
            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]

            pairs += [[x1, x2]]
            labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")


def make_train_set_samples(x, y) -> (np.array, np.array):
    num_classes = max(y) + 1
    print(num_classes)
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    X = []
    Y = []

    for idx1 in range(len(x)):
        x1 = x[idx1]
        label1 = y[idx1]

        for _ in range(5):
            sample = []
            labels = []

            for label2 in range(num_classes):
                idx2 = random.choice(digit_indices[label2])
                x2 = x[idx2]
                sample += [[x1, x2]]

                if label1 == label2:
                    labels += [0]
                else:
                    labels += [1]

            X.append(sample)
            Y.append(labels)

    return np.array(X), np.array(Y).astype("float32")


def make_train_set_samples_with_snippets(x, y, snippets_x, snippets_y) -> (np.array, np.array):
    num_classes = max(y) + 1
    print(num_classes)
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    X = []
    Y = []

    for idx1 in range(len(x)):
        x1 = x[idx1]
        label1 = y[idx1]

        for _ in range(5):
            sample = []
            labels = []

            for label2 in range(num_classes):
                idx2 = random.choice(digit_indices[label2])
                x2 = x[idx2]
                sample += [[x1, x2]]

                if label1 == label2:
                    labels += [0]
                else:
                    labels += [1]

            X.append(sample)
            Y.append(labels)

        sample_with_snippets = []
        labels_with_snippets = []

        for label2 in range(num_classes):
            snippet = snippets_x[label2]
            sample_with_snippets += [[x1, snippet]]

            if label1 == label2:
                labels_with_snippets += [0]
            else:
                labels_with_snippets += [1]

        X.append(sample_with_snippets)
        Y.append(labels_with_snippets)

    return np.array(X), np.array(Y).astype("float32")


def make_test_set_pairs(x_test, y_test, x_snippets, y_snippets) -> (np.array, np.array):
    pairs = []
    labels = []

    for i in range(len(x_test)):
        x1 = x_test[i]
        y1 = y_test[i]

        for j in range(len(x_snippets)):
            x2 = x_snippets[j]
            y2 = y_snippets[j]

            pairs += [[x1, x2]]
            if y1 == y2:
                labels += [0]
            else:
                labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")


def make_test_set_samples(x_test, y_test, x_snippets, y_snippets) -> (np.array, np.array):
    X_test = []
    Y_test = []

    for i in range(len(x_test)):
        sample_test = []
        labels_test = []

        x1 = x_test[i]
        y1 = y_test[i]

        for j in range(len(x_snippets)):
            x2 = x_snippets[j]
            y2 = y_snippets[j]

            sample_test += [[x1, x2]]
            if y1 == y2:
                labels_test += [0]
            else:
                labels_test += [1]

        X_test.append(sample_test)
        Y_test.append(labels_test)

    return np.array(X_test), np.array(Y_test).astype("float32")


def make_original_test_ts_pairs(x_test, x_snippets):
    pairs = []

    for i in range(len(x_test)):
        x1 = x_test[i]
        for j in range(len(x_snippets)):
            x2 = x_snippets[j]
            pairs += [[x1, x2]]

    return np.array(pairs)


def make_original_test_ts_samples(x_test, x_snippets) -> np.array:
    X_test = []

    for i in range(len(x_test)):
        sample_test = []

        x1 = x_test[i]

        for j in range(len(x_snippets)):
            x2 = x_snippets[j]
            sample_test += [[x1, x2]]

        X_test.append(sample_test)

    return np.array(X_test)


def make_results_table(similarity_score_list, test_pairs_labels, y_test, snippets_num):
    test_subs_count = len(y_test)
    similarity_score_table = [similarity_score_list[snippets_num * i: snippets_num * (i + 1)] for i in range(test_subs_count)]

    results_table = []

    for i in range(test_subs_count):
        min_similarity_score = np.min(similarity_score_table[i])
        min_similarity_score_idx = np.argmin(similarity_score_table[i])
        predict_pair_label = str(test_pairs_labels[i * snippets_num + min_similarity_score_idx])
        real_label = y_test[i]
        if real_label == -1:
            real_label = '1.0'
        else:
            real_label = '0.0'
        results_table.append([min_similarity_score, predict_pair_label, real_label])

    df_results_table = pd.DataFrame(results_table)
    df_results_table.columns = ['min_similarity_score', 'predict_label', 'real_label']

    return df_results_table


def calculate_threshold(normal_scores, n_percentile):
    threshold = normal_scores[int(np.ceil(n_percentile / 100 * len(normal_scores)) - 1)]
    return threshold.item()


def find_subs_similarity_scores(similarity_scores, snippets_num):
    min_scores = []
    for i in range(0, len(similarity_scores), snippets_num):
        min_score = similarity_scores[i][0]
        for j in range(snippets_num - 1):
            min_score = min(min_score, similarity_scores[i + j + 1][0])

        min_scores.append(min_score)

    return min_scores


def find_anomaly_regions(subs_similarity_scores, threshold):
    bigger_threshold_idxs = [i for i, e in enumerate(subs_similarity_scores) if e >= threshold]
    start_groups = []

    for i in range(1, len(bigger_threshold_idxs)):
        curr_elem = bigger_threshold_idxs[i - 1]
        next_elem = bigger_threshold_idxs[i]

        if next_elem - curr_elem > 1:
            start_groups.append(curr_elem)

    if len(bigger_threshold_idxs) == 0:
        return []
    start_groups.append(next_elem)

    end_groups = []
    end_groups.append(bigger_threshold_idxs[0])

    for i in range(len(bigger_threshold_idxs) - 1):
        curr_elem = bigger_threshold_idxs[i]
        next_elem = bigger_threshold_idxs[i + 1]

        if next_elem - curr_elem > 1:
            end_groups.append(next_elem)

    return list(zip(end_groups, start_groups))


def save_snn_params(nn_type, epochs, batch_size, margin, optimizer, file_path):
    snn_params = {
        'nn_type': nn_type,
        'epochs': epochs,
        'batch_size': batch_size,
        'margin': margin,
        'optimizer': optimizer,
    }

    with open(file_path, 'w') as outfile:
        json.dump(snn_params, outfile, indent=4)


def save_model_summary(model, file_path):
    with open(file_path, 'w') as outfile:
        model.summary(print_fn=lambda x: outfile.write(x + '\n'))


def save_snn_history(history, file_path):
    hist_df = pd.DataFrame(history.history)
    with open(file_path, mode='w') as f:
        hist_df.to_json(f, indent=4)


def load_csv_file(file_path):
    data = pd.read_csv(file_path, header=None)
    return data.iloc[:, 0].values.tolist()


def get_real_anomaly_annotation(input_files, input_ts_lengths, train_ts_lengths):
    real_anomaly_annotation = []
    test_ts_start_idx = 0

    for i in range(len(input_files)):
        annotation_file = input_files[i].split('.')[0] + '_annotation.csv'
        ts_annotation = load_csv_file(os.path.join(config.ANNOTATION_DIR, annotation_file))
        test_ts_annotation = [x for x in ts_annotation if x >= train_ts_lengths[i]]
        test_len_ts = input_ts_lengths[i] - train_ts_lengths[i]
        updated_test_ts_annotation = [x - test_len_ts + test_ts_start_idx for x in test_ts_annotation]
        test_ts_start_idx = test_ts_start_idx + test_len_ts
        real_anomaly_annotation.extend(updated_test_ts_annotation)

    return real_anomaly_annotation


def find_predict_anomaly_annotation(subs_similarity_scores, threshold):
    subs_count = len(subs_similarity_scores)
    predict_anomaly_annotation = np.array([1] * subs_count)

    predict_anomaly_annotation[np.where(np.array(subs_similarity_scores) >= threshold)[0]] = -1

    return predict_anomaly_annotation.tolist()


def find_top_k_predicted_anomalies(subs_similarity_scores, real_anomalies_num, m):
    desc_sort_scores = np.flip(np.sort(subs_similarity_scores))
    desc_sort_scores_idx = np.flip(np.argsort(subs_similarity_scores))

    predict_anomalies_idx = []
    predict_anomalies_idx.append(desc_sort_scores_idx[0])
    predict_anomalies_num = 1
    i = 1

    while predict_anomalies_num < real_anomalies_num:
        predict_anomaly_idx = desc_sort_scores_idx[i]

        is_anomaly = True
        for j in range(len(predict_anomalies_idx)):
            if ((predict_anomalies_idx[j] - m < predict_anomaly_idx) & (predict_anomaly_idx < predict_anomalies_idx[j] + m)):
                is_anomaly = False
                break

        if is_anomaly:
            predict_anomalies_idx.append(predict_anomaly_idx)
            predict_anomalies_num = predict_anomalies_num + 1

        i = i + 1

    sort_predict_anomalies_idx = np.sort(predict_anomalies_idx)

    return sort_predict_anomalies_idx


def calculate_accuracy_metrics(real_annotation, predict_annotation):
    return


def calculate_precision(actual, predicted, m):
    TP = 0

    for predict_idx in predicted:
        is_hit = False

        for true_idx in actual:
            if ((true_idx - m < predict_idx) & (predict_idx < true_idx + m)):
                is_hit = True
                break

        if is_hit:
            TP = TP + 1

    return TP / len(actual)


def save_metrics(accuracy_metrics, file_path):
    with open(file_path, 'w') as file:
        w = csv.DictWriter(file, accuracy_metrics.keys())
        w.writeheader()
        w.writerow(accuracy_metrics)


def write_dataset(set, dir, file_name):
    with open(os.path.join(dir, file_name), 'w') as outfile:
        write = csv.writer(outfile)
        write.writerows(set)
