# Anomaly-monitoring-diplom-maga
Дипломный проект магистратуры

mDISSID: https://github.com/KraevaYA/mDiSSiD разработка взята с репозитория и применена в текущем проекте в качестве метода обнаружения аномалий через replay-service (mDiSSiD: Discord, Snippet, and Siamese Neural Network-based Detector of multivariate anomalies). 

## Пример использования
### Установка зависимостей
pip install -r requirements.txt


### Проверка статуса

 Открыть браузер и проверить Grafana: http://localhost:3000
 Логин: admin, пароль: admin
 
 Добавьте источник данных Prometheus:
    URL: http://prometheus:9090


# Запуск проекта

## Шаг 1. Запустить Docker-инфраструктуру

Из корня проекта:

docker-compose start
docker-compode up


## Шаг 2. Отдельно запустить replay-сервисы через main_service.py

Все команды выполняются из корня проекта:

```powershell
cd F:\Documents\GitHub\Anomaly-monitoring-diplom-maga

# запуск всех трех моделей
python .\node_replay_service\main_service.py `
  --models damp,mdissid,lstm `
  --prepared_dir .\data\prepared_nodes `
  --max_nodes 25 `
  --tick_seconds 0.2 `
  --rows_per_tick 1 `
  --thresholds_dir .\node_replay_service\thresholds `
  --damp_port 8011 `
  --mdissid_port 8010 `
  --lstm_port 8002 `
  --repo_snn_dir .\mdissid_work\mDiSSiD-main\src\SNN `
  --datasets_root .\mdissid_work\mDiSSiD-main\datasets\SNN_datasets `
  --results_root .\mdissid_work\mDiSSiD-main\SNN_results `
  --baseline_dataset_name node001_multivariate_2641_20_10_2 `
  --lstm_output_dir .\lstm_work\lstm_output `
  --lstm_baseline_node_name node001 `
  --lstm_train_if_missing

## Только DAMP на 25 узлах

python .\node_replay_service\main_service.py `  
  --models damp `  
  --prepared_dir .\data\prepared_nodes `  
  --max_nodes 25 `  
  --tick_seconds 0.2 `  
  --rows_per_tick 1 `  
  --thresholds_dir .\node_replay_service\thresholds

## Только mDiSSiD на 25 узлах

python .\node_replay_service\main_service.py `  
  --models mdissid `  
  --prepared_dir .\data\prepared_nodes `  
  --max_nodes 25 `  
  --tick_seconds 0.2 `  
  --rows_per_tick 1 `  
  --thresholds_dir .\node_replay_service\thresholds `  
  --repo_snn_dir .\mdissid_work\mDiSSiD-main\src\SNN `  
  --datasets_root .\mdissid_work\mDiSSiD-main\datasets\SNN_datasets `  
  --results_root .\mdissid_work\mDiSSiD-main\SNN_results `  
  --baseline_dataset_name node001_multivariate_2641_20_10_2

## DAMP + mDiSSiD вместе на 25 узлах

python .\node_replay_service\main_service.py `  
  --models damp,mdissid `  
  --prepared_dir .\data\prepared_nodes `  
  --max_nodes 25 `  
  --tick_seconds 0.2 `  
  --rows_per_tick 1 `  
  --thresholds_dir .\node_replay_service\thresholds `  
  --repo_snn_dir .\mdissid_work\mDiSSiD-main\src\SNN `  
  --datasets_root .\mdissid_work\mDiSSiD-main\datasets\SNN_datasets `  
  --results_root .\mdissid_work\mDiSSiD-main\SNN_results `  
  --baseline_dataset_name node001_multivariate_2641_20_10_2
