# A single cell RNA-Seq based aging clock for human neurons

### Deep learning models for predicting age

Aging is a biological process that is still a conundrum yet to solve. Researchers have already worked out age can be predicted from external signs like wrinkles, sagging cheeks, etc. But, the internal human age tells a different story. Can we predict human age given the gene expression data of a patient? That is the question we sought to answer.

Our data comprises single cell RNA-seq expression from 17658 genes x 189423 cells, from 69 patients. Initially we conducted a simple linear regression on single cell RNA-seq data to see how well we can conduct regression analysis and predicted ages with a mean absolute error of 11.21.

Our first round of results run on simple Multi-layer Perceptrons with tuned hyper-parameters did not get us the results we expected as it turned out to be slightly worse than the linear regression model. 

We implemented Non-negative Matrix Factorization, Standard (Gaussian) Variational Autoencoder, Poisson Variational Autoencoder and De-noising Criterion with Variational Autoencoder. And as we expected Poisson VAE got us much better results than any of the techniques we implemented! This is mainly because the likelihood function (here, poisson) helped in modeling the huge sparse data much better than others.

#### Contributors
Mirudhula Mukundan, Qiao Su
