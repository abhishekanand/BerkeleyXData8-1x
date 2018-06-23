"""
# https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-python?tabs=cntkpy251

PreReq :
1. Anacoda 

CPU Only 

   python
   pip install https://cntk.ai/PythonWheel/CPU-Only/cntk-2.5.1-cp36-cp36m-win_amd64.whl
   3 python -m pip install --upgrade pip
   4 ls -l
   5 ls
   6 python -m cntk.sample_installer
   7 import cntk


pip install ipykernel
pip install psyplot
pip install psyplot-gui

GPU 


   8 python
   9 pip install https://cntk.ai/PythonWheel/GPU/cntk_gpu-2.5.1-cp36-cp36m-win_amd64.whl

Verify CNTK 

python -c "import cntk; print(cntk.__version__)"

# Getting Started  :
https://cntk.ai/pythondocs/gettingstarted.html#first-basic-use


# Settin Up GPU Target 

from cntk.device import try_set_default_device, gpu
try_set_default_device(gpu(0))

  python .\simplenet.py

"""


from __future__ import print_function
import numpy as np
import cntk as C
from cntk.learners import sgd
from cntk.logging import ProgressPrinter
from cntk.layers import Dense, Sequential

def generate_random_data(sample_size, feature_dim, num_classes):
     # Create synthetic data using NumPy.
     Y = np.random.randint(size=(sample_size, 1), low=0, high=num_classes)

     # Make sure that the data is separable
     X = (np.random.randn(sample_size, feature_dim) + 3) * (Y + 1)
     X = X.astype(np.float32)
     # converting class 0 into the vector "1 0 0",
     # class 1 into vector "0 1 0", ...
     class_ind = [Y == class_number for class_number in range(num_classes)]
     Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
     return X, Y

def ffnet():
    inputs = 2
    outputs = 2
    layers = 2
    hidden_dimension = 50

    # input variables denoting the features and label data
    features = C.input_variable((inputs), np.float32)
    label = C.input_variable((outputs), np.float32)

    # Instantiate the feedforward classification model
    my_model = Sequential ([
                    Dense(hidden_dimension, activation=C.sigmoid),
                    Dense(outputs)])
    z = my_model(features)

    ce = C.cross_entropy_with_softmax(z, label)
    pe = C.classification_error(z, label)

    # Instantiate the trainer object to drive the model training
    lr_per_minibatch = C.learning_parameter_schedule(0.125)
    progress_printer = ProgressPrinter(0)
    trainer = C.Trainer(z, (ce, pe), [sgd(z.parameters, lr=lr_per_minibatch)], [progress_printer])

    # Get minibatches of training data and perform model training
    minibatch_size = 25
    num_minibatches_to_train = 1024

    aggregate_loss = 0.0
    for i in range(num_minibatches_to_train):
        train_features, labels = generate_random_data(minibatch_size, inputs, outputs)
        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        trainer.train_minibatch({features : train_features, label : labels})
        sample_count = trainer.previous_minibatch_sample_count
        aggregate_loss += trainer.previous_minibatch_loss_average * sample_count

    last_avg_error = aggregate_loss / trainer.total_number_of_samples_seen

    test_features, test_labels = generate_random_data(minibatch_size, inputs, outputs)
    avg_error = trainer.test_minibatch({features : test_features, label : test_labels})
    print(' error rate on an unseen minibatch: {}'.format(avg_error))
    return last_avg_error, avg_error

np.random.seed(98052)
ffnet()


"""
PS C:\Users\abhanand\Documents> python .\simplenet.py
Selected GPU[0] Quadro K420 as the process wide default device.
-------------------------------------------------------------------
Build info:

                Built time: Apr 13 2018 22:35:30
                Last modified date: Fri Apr 13 17:13:20 2018
                Build type: Release
                Build target: GPU
                With ASGD: yes
                Math lib: mkl
                CUDA version: 9.0.0
                CUDNN version: 7.0.5
                Build Branch: HEAD
                Build SHA1: 11c82c8019b78323b4f5169a2d8c90fba5ae49da
                MPI distribution: Microsoft MPI
                MPI version: 7.0.12437.6
-------------------------------------------------------------------
 average      since    average      since      examples
    loss       last     metric       last
 ------------------------------------------------------
Learning rate per minibatch: 0.125
    0.685      0.685       0.52       0.52            25
    0.665      0.656      0.507        0.5            75
    0.662       0.66      0.406       0.33           175
    0.604      0.553      0.347      0.295           375
    0.547      0.493       0.25       0.16           775
    0.477       0.41      0.183      0.117          1575
    0.399      0.323      0.141     0.0994          3175
    0.332      0.265      0.115       0.09          6375
     0.28      0.228     0.0964     0.0775         12775
    0.248      0.215     0.0874     0.0784         25575
 error rate on an unseen minibatch: 0.0
 """
