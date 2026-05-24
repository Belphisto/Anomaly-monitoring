# coding: utf-8

import numpy as np
import os

import config
import matrix_profile
import top_k_discords
import snippets
import snippets_anomalies
import plots
import utils


def main():

    args = utils.parse_args()
    
    multi_train_ts, multi_test_ts, train_label, test_label, ts_lengths = utils.split_ts(args.input_files, args.train_lengths)

    n = multi_train_ts.shape[0] # length of multivariate time series
    d = multi_train_ts.shape[1] # number of dimensions (the last column is label)
    N = n - args.m + 1
    
    print(multi_train_ts)
    
    print(f"The length of the train time series = {n}")
    print(f"The number of dimensions in time series = {d}")

    dataset_name = utils.get_dataset_name(args.input_files, n, args.m, args.l, args.snippets_num)
    snn_dataset_dir = os.path.join(config.SNN_DATASETS_DIR, dataset_name)
    utils.create_directory(snn_dataset_dir)
    
    print(dataset_name)

    plots_dir = os.path.join(config.PLOTS_DIR, dataset_name)
    utils.create_directory(plots_dir)
    image_plots_dir = os.path.join(plots_dir, 'images')
    utils.create_directory(image_plots_dir)
    data_plots_dir = os.path.join(plots_dir, 'data')
    utils.create_directory(data_plots_dir)

    # plot time series
    plots.plot_multivariate_ts(multi_train_ts, n, d, os.path.join(image_plots_dir, 'train_multivariate_ts.png'), title='Multivariate Train Time Series')
    
    multi_snippets = {}
    common_anomalies_annotation = [0]*N

    for i in range(d):
        # 1. find matrix profile
        
        print(f"Start to preprocess time series {i}")
        snn_dimension_dataset_dir = os.path.join(snn_dataset_dir, str(i))
        utils.create_directory(snn_dimension_dataset_dir)
        
        print("1. Start to find Matrix Profile")
        mp = matrix_profile.find_mp(multi_train_ts[:,i], args.m)
        plots.plot_ts(mp['mp'], len(mp['mp']), os.path.join(image_plots_dir, f"matrix_profile_{i}.png"), title='Matrix Profile')
        utils.write_dataset(np.array(mp['mp']).reshape(-1,1), data_plots_dir, f"matrix_profile_{i}.csv")
        print("Matrix Profile is computed\n")
        
        # 2. find discords in time series
        print("2. Start to find discords in the time series")
        discords_num = int(np.ceil(args.alpha*N))
        ts_discords = top_k_discords.find_discords(mp, args.m, discords_num)
        discords_idx = list(ts_discords['discords'])
        discords_idx.sort()
        discords_annotation = top_k_discords.construct_discords_annotation(discords_idx, N, args.m)
        plots.plot_discords(multi_train_ts[:,i], mp['mp'], ts_discords['discords'], n, args.m, N, discords_num, os.path.join(image_plots_dir, f"discords_{i}.png"))
        utils.write_discords(ts_discords, data_plots_dir, f"discords_{i}.json")
        utils.write_dataset(np.array(discords_annotation).reshape(-1,1), data_plots_dir, f"discords_annotation_{i}.csv")
        print(f"{discords_num} Discords are founded in the time series\n")
        
        # 3. find snippets in the time series
        print("3. Start to find snippets in the time series")
        if (config.SNIPPET_FIND_WITH_OPTIMIZATION):
            ts_snippets = snippets.find_snippets_with_optimization(multi_train_ts[:,i], args.m, args.l, args.snippets_num)
        else:
            ts_snippets = snippets.find_snippets_without_optimization(np.array(multi_train_ts[:,i]), args.m, args.l, args.snippets_num)
        multi_snippets['ts'+str(i)] = ts_snippets

        profiles_curve = snippets.find_profiles_curve(ts_snippets['profiles'], args.snippets_num)

        plots.plot_snippets(multi_train_ts[:,i], ts_snippets, n, args.m, args.snippets_num, os.path.join(image_plots_dir, f"snippets_{i}.png"))
        utils.write_snippets(ts_snippets, data_plots_dir, 'snippets.json')
        plots.plot_profiles(ts_snippets['profiles'], len(ts_snippets['profiles'][0]), args.snippets_num, os.path.join(image_plots_dir, f"mpdist_profiles_{i}.png"))
        utils.write_dataset([profile.tolist() for profile in ts_snippets['profiles'].T], data_plots_dir, f"mpdist_profiles_{i}.csv")
        print("3. Snippets are founded in the time series\n")
        
        # 4. find the snippets anomalies
        print("4. Start to find snippets anomalies in the time series")
        moving_max_profiles = []
        w = args.m//2+1 # window width
        
        for j in range(args.snippets_num):
            moving_max_profiles.append(utils.moving_max(ts_snippets['profiles'][j], w))
        
        if (config.SNIPPETS_ANOMALY_METHOD == 'IsolationForest'):
            print(config.SNIPPETS_ANOMALY_METHOD)
            max_mpdist_regimes = snippets.find_mpdist_regimes(ts_snippets['regimes'], moving_max_profiles, args.snippets_num)
            max_regimes_profiles = snippets.find_regimes_profiles(max_mpdist_regimes, N, args.snippets_num)
            #ts_snippets_anomalies = snippets_anomalies.find_snippets_anomalies_IF(max_regimes_profiles, args.snippets_num)
            ts_snippets_anomalies = snippets_anomalies.find_snippets_anomalies_IF(moving_max_profiles, args.snippets_num)
        else:
        # if KNN
            print(config.SNIPPETS_ANOMALY_METHOD)
            max_mpdist_regimes = snippets.find_mpdist_regimes(ts_snippets['regimes'], moving_max_profiles, args.snippets_num)
            max_mpdist_all_regimes = snippets.find_mpdist_all_regimes(ts_snippets['regimes'], moving_max_profiles, args.snippets_num)
            ts_snippets_anomalies = snippets_anomalies.find_snippets_anomalies_KNN(max_mpdist_all_regimes, ts_snippets['indices'], N, args.snippets_num)

        print(len(ts_snippets_anomalies))
        
        snippets_anomalies_annotation = snippets_anomalies.construct_snippets_anomalies_annotation(max_mpdist_regimes, ts_snippets_anomalies, ts_snippets['indices'], N+1, args.snippets_num, args.m)
        plots.plot_annotation(multi_train_ts[:,i], train_label, snippets_anomalies_annotation, len(snippets_anomalies_annotation), os.path.join(image_plots_dir, f"snippets_anomalies_annotation_{i}.png"), title="Snippets anomalies annotation")
        utils.write_dataset(np.array(snippets_anomalies_annotation).reshape(-1,1), data_plots_dir, f"snippets_anomalies_annotation_{i}.csv")
        print("4. Snippets anomalies are founded in the time series\n")

        # 5. find the anomaly annotation
        print("5. Start to find anomaly annotation")
        print(len(discords_annotation))
        print(len(snippets_anomalies_annotation))
        #snippets_anomalies_annotation = [1]*len(snippets_anomalies_annotation)
        
        anomalies_annotation = snippets_anomalies.construct_anomalies_annotation(discords_annotation, snippets_anomalies_annotation[:len(discords_annotation)])
        plots.plot_annotation(multi_train_ts[:,i], train_label, anomalies_annotation, len(anomalies_annotation), os.path.join(image_plots_dir, f"anomalies_annotation_{i}.png"), title="Anomalies annotation")
        utils.write_dataset(np.array(anomalies_annotation).reshape(-1,1), data_plots_dir, f"anomalies_annotation_{i}.csv")
        print("5. Anomaly annotation is founded\n")

        
        common_anomalies_annotation = np.sum([common_anomalies_annotation, anomalies_annotation], axis=0)
        
    common_anomalies_annotation = list(np.where(np.array(common_anomalies_annotation) < d, -1, 1))
        
    for i in range(d):
        
        snn_dimension_dataset_dir = os.path.join(snn_dataset_dir, str(i))
             
        # 6. generate the train and test sets for Siamese Neural Network
        print("6. Start to generate the train and test sets for Siamese Neural Network")
        subsequences_labels = utils.label_subsequences(multi_snippets['ts'+str(i)], common_anomalies_annotation, len(anomalies_annotation), args.m, args.snippets_num) #ts_snippets['fractions'].tolist())
        utils.write_dataset(np.array(subsequences_labels).reshape(-1,1), data_plots_dir, f"subsequences_labels_{i}.csv")

        sets = []
        train_set, test_set, train_set_idx, test_set_idx = utils.generate_neural_network_datasets(multi_train_ts[:,i], subsequences_labels, args.m, args.test_normal_set_size)
        sets.append(train_set)
        sets.append(test_set)
        print("6. Generation of the train and test sets for Siamese Neural Network finished\n")

        # 7. create the snippet set
        print("7. Start to create the snippet set")
        snippets_set = utils.create_snippets_set(multi_snippets['ts'+str(i)]['snippets'].tolist(), args.snippets_num)
        sets.append(snippets_set)
        print("7. The snippet set is created\n")

        # 8. create the statistics tables for train and test sets
        print("8. Start to create the statistics tables")
        train_statistics, test_statistics = utils.create_statistics_tables(train_set_idx, test_set_idx, mp['mp'], profiles_curve)
        sets.append(train_statistics)
        sets.append(test_statistics)
        print("8. Statistics tables are created\n")

        # 9. write datasets
        print("9. Start to write datasets for SNN")
        for j in range(len(sets)):
            utils.write_dataset(sets[j], snn_dimension_dataset_dir, config.OUTFILE_NAMES[j])
        print("9. Datasets for SNN are written and saved\n")

        # 10. write the original test time series and label
        print("10. Start to write the original test time series and label for SNN")
        utils.write_dataset(multi_test_ts[:,i].reshape(-1,1), snn_dimension_dataset_dir, config.OUTFILE_NAMES[-3])
        utils.write_dataset(test_label.reshape(-1,1), snn_dimension_dataset_dir, config.OUTFILE_NAMES[-2])
        print("10. The original test time series and label are written and saved\n")
        
        # 11. write the input parameters into the json file
        print("11. Start to write the input parameters into the json file")
        utils.save_input_params(args, ts_lengths, n, d, snn_dimension_dataset_dir, config.OUTFILE_NAMES[-1])
        print("11. The input parameters ate written and saved\n")
        
        print(f"Finish to preprocess time series {i}")

    plots.plot_multi_snippets(multi_train_ts, multi_snippets, train_label, n, d, args.m, args.snippets_num, os.path.join(image_plots_dir, 'multi_snippets.png'))


if __name__ == '__main__':
    main()
