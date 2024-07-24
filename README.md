# Multi-kernel ridge regression (MKRR)

**MKRR** is a multi-kernel extension of kernel ridge regression (KRR) that integrates genotype data and gene expression data through a multi-kernel learning (MKL) strategy for genomic prediction. 

## Tutorial and Examples

We implemented MKRR in Python. Dependencies: python > 3.6.

We provided example code and toy datasets to illustrate how to use MKRR for hyperparameter optimization and genomic prediction. Please check MKRR.py to see how to run MKRR on the toy example we provided in the example_data directory. 

### Prepare files

The prepare files need to be placed in the `example_data` folder and include the following five files：

1、`train_geno.txt`: The genotype file for the training set individuals. The first column is the individual IDs, and from the second column onward, each column represents a SNP marker (encoded as 0, 1, or 2). 

2、`train_TPM.txt`: The gene expression file for the training set individuals. The first column is the individual IDs, and from the second column onward, each column represents a gene.

3、`train_y.txt`: The phenotype file for the training set individuals. The first column is the individual IDs, and the second column is the phenotype values.

4、`test_geno.txt`: The genotype file for the test set individuals. The first column is the individual number, and from the second column onward, each column represents a SNP marker (encoded as 0, 1, or 2). 

5、`test_TPM.txt`: The gene expression file for the test set individuals. The first column is the individual IDs, and from the second column onward, each column represents a gene.

### Running command

Before running the program, the users needs to install the required packages (psutil, numpy, pandas, scikit-learn, scikit-optimize, scipy). Then, place the software and the `example_data` folder in the same directory. Enter the current directory and run the program by typing the command `python MKRR.py`. For example:

```sh
cd path/to/your/directory
python MKRR.py
```

### output files

The output files will be stored in the `results` folder and include `Best_params.txt` and `Predicted_test_y.txt`.

1、`Best_params.txt`: The optimal hyperparameters determined by the Bayesian optimization algorithm.

2、`Predicted_test_y.txt`: The predicted values for the test set individuals obtained by fitting the model using the optimal hyperparameters. The file contains two columns: individual IDs and their corresponding predicted values.



## Other Notes on the Software

- For the training set data, please make sure that the order of individuals (rows) is consistent across the genotype file (`train_geno.txt`), the gene expression file (`train_TPM.txt`), and the phenotype file (`train_y.txt`).
- For the test set data, please make sure that the order of individuals (rows) in the genotype file (`test_geno.txt`) matches the order of individuals in the gene expression file (`test_TPM.txt`).



## QUESTIONS AND FEEDBACK

For questions or concerns with MKRR software, please contact xwangchnm@163.com.

We welcome and appreciate any feedback you may have with our software and/or instructions.
