# Quasi-Recurrent Neural Networks

A TensorFlow implementation of a proposed
alternative to RNN based on convolutions described in the paper
[Quasi-Recurrent Neural Networks](https://arxiv.org/pdf/1611.01576.pdf).

See my [report](../master/qrnn/report.pdf) for further details.
Summaries from the training of the LSTM based model and the QRNN model
can be found as tfrecords files in the directories 'lstm_summary' and
'qrnn_summary' and can be viewed using Tensorboard.

The actual implementation of the QR layer is in
[qr_imdb/qrnn.py](../master/qrnn/qr_imdb/qrnn.py)

### How to run training and evaluation

Please note, that the training script need an additional file - 'glove.840B.300d.txt'
to create a word embedding matrix. We recommend to download the file
from https://nlp.stanford.edu/projects/glove/ and extract it separately.

To run training and evaluation (10 epochs) of the QRNN model, execute the following:

    $ python main.py --glovepath /path/to/glove.840B.300d.txt

If the argument glovepath is not provided, the script will download and extract
the needed file. However, be warned, it takes a while and I did not bother
to include a progress bar for the download.

If you would like to run the training and evaluation with a different model
from 'qr_imdb/models', see

    $ python main.py --help

for instructions.

The preprocessed IMDb dataset is included as a part of the repository,
because extracting and processing cca 100000 individual text files
takes a while. If you would like to run the preprocessing script yourself,
it assumes that the dataset from
[http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)
has been extracted to 'data/aclImdb'. To run the script, execute

    $ python ./utils/imdb_preprocessing.py

### LaTex template

The template used can be found [here](http://www.latextemplates.com/template/journal-article).
