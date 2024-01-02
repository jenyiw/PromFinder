# Predicting transcriptional start sites

The PromFinder is an attempt at detecting transcriptional start sites using convolutional neural networks, random forest and Support Vector Machines. This was based on ideas in literature such as the TSSFinder (https://academic.oup.com/bib/article/22/6/bbab198/6287335). The prediction was done based purely on DNA structural properties and checked against the FANTOM database. 

## Results
The convolutional neural network and support vector machine performed well when tested against randomly generated background sequences, but did not perform better than random when tested with randomly sampled DNA sequences as the negative sample. This may be due to the higher GC content of transcriptional start sites, which makes it distinct from random sequences but not from sampled sequences. 

## Getting Started

### Prerequisites

The following packages are required: 
1. NumPy 1.24.3
2. scikit-learn 1.3.2
3. torch 2.0.1
4. Pandas 2.1.1

## Running the code

To run the code, run 'main_without_GUI.py'


## Authors

* **Jenyi Wong**
* **Linda Zhou**
* **Ziyang**
* **Felix Zheng**
  
## Acknowledgments

* Prof. Joel McManus
