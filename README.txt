Platform: Python3.7 + Tensorflow 2.6.0 


1. Run "main_MAE.py" file to pretrain feature extractors for AF and DAPI images, respectively. A random masked mechanism is employed, where a mask of random size is applied to each patch at a random location during MAE training in each epoch. Early stop is used to define the model's convergence.

2. Run "main_task1" file to train the MAS model for the staining task of CD3, CD20, FOXP3, and PD1 biomarkers. 

3. Run "main_task2" file to transfer the MAS model for the adaption of CD8, CD163, and PDL1 biomarkers. 

4. Use "test" file to implement a quick virtual staining for label-free whole slides with trained model.

Please cite paper "Generative AI-Powered Virtual Multiplex Staining from Label-Free Fluorescence Images for Precision Medicine" if the code is helpful for your work. If you have any questions, please feel free to contact me through zixia@stanford.edu