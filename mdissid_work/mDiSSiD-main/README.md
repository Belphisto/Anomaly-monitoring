# mDiSSiD: Discord, Snippet, and Siamese Neural Network-based Detector of multivariate anomalies
This repository is related to a novel method, namely mDiSSiD (Discord, Snippet, and Siamese Neural Networket-based Detector of multivariate anomalies), for detecting anomalous subsequences of multivariate streaming time series. mDiSSiD is authored by Yana Kraeva (kraevaya@susu.ru), South Ural State University, Chelyabinsk, Russia. The repository contains the mDiSSiD's source code (in Python), accompanying datasets, and experimental results. Please cite an article that describes mDiSSiD as shown below.

The new method mDiSSiD generalizes the DiSSiD (Discord, Snippet, and Siamese Neural Network-based Detector of anomalies) method proposed by the author earlier [https://doi.org/10.14529/cmse230304] for detecting anomalies in a one-dimensional streaming time series to the multidimensional case. The mDiSSiD method employs the time series discord concept (a subsequence with the most dissimilar nearest neighbor). Multivariate discord refers to the $N$-dimensional subsequence of a $d$-dimensional time series (where $1 \leqslant N \leqslant d$), which is the most dissimilar to all other subsequences of $N$-dimensional time series obtained by composing all the possible combinations of $d$ series of $N$ [https://doi.org/10.1137/1.9781611977653.CH77]. Anomaly detection is implemented through a deep learning model based on the Siamese neural network architecture. The mDiSSiD model is an ensemble of $d$ modifications of the DiSSiD model.

# Citation
```
@article{Kraeva2024,
 author    = {Yana A. Kraeva},
 title     = {Deep Learning Method for Anomaly Detection in Streaming Multivariate Time Series},
 journal   = {Bulletin of the South Ural State University. Series: Computational Mathematics and Software Engineering},
 volume    = {13},
 number    = {4},
 pages     = {35-52},
 year      = {2024},
 doi       = {10.14529/cmse240403},
 url       = {https://doi.org/10.14529/cmse240403}
}
```
# Acknowledgement
This work was financially supported by the Russian Science Foundation (grant no. 23-21-00465).
