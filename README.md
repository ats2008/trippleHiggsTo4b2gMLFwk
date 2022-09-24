# trippleHiggsTo4b2gMLFwk

ML Framework for developing models for HHH-->4b 2g 


## Setup

We use [`conda`](https://conda.io/projects/conda/en/latest/index.html) for managing the pacakges used for the development.
For instantiation of the environment one can use the `yaml` config provided

```bash 
conda env create -f  env.yaml
```

We use root files as entry point for data via. [Uproot](https://uproot.readthedocs.io/en/latest/index.html). The workflow is based on PyTorch framework. 


## Development Workflow

Setting up a remote jupyter notebook session [ set it up of a machine with GPU support for quicker dev cycle ]
```bash 
<local>  $ <login to server >
<server> $ cd ~/work/g2Net/
<server> $ jupyter notebook --no-browser --port=<PORT>  #change port number if the posrt is busy 
```

Connecting to remote jupyter notebook session
```bash
<local> $ ssh -L 8080:localhost:<PORT> <REMOTE_USER>@<REMOTE_HOST>
```
open [http://localhost:8080/](http://localhost:8080/) to go to the Jupyter Notebook web interface. If password is asked , give the token shown at the end of link shown while launching the notebook in server


Launch [tensorbord](https://www.tensorflow.org/tensorboard) for monitoring the training
```bash
tensorboard --logdir <LOGDIR> --port=<PORT>  >& /dev/null &
#tensorboard --logdir workarea/ml/attention4HHH/checkpoints/trippleHiggsVsQCD/lightning_logs/ --port=8008 >& /dev/null &
```


