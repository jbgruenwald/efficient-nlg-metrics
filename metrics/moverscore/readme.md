This directory contains a modified version of MoverScore (https://github.com/AIPHES/emnlp19-moverscore).

The scorer was changed into a class to make replacing of the model easier.
The class was also changed to be able to make calculations on a gpu like xmoverscore is able.
The multiprocessing import was changed into a torch.multiprocessing and set_start_method('spawn') is executed after the import to make pytorch able to use multiprocessing on gpu.