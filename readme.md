# Road Extraction with D-Link Net

## Usage

### To Train:

Run `train_slim_model.py`

Options:

`--data_dir=<path>`

    Path to training data.
    
`--summary_dir=<path>`

    Save summary to specified path.

`--save_dir=<path>` 

    Save model to specified path.
    By default the model will be saved under <path>/model_<time_string>/
    
`--no_append`

    If set, model will be saved directly under save_dir. No sub directory will be made.
    
`--resume_dir=<path>`

    If set, resume training the model from a previous checkpoint

`--CKPT_RES50=<path>`

    Path to ResNet 50 pre-trained model.
    
`--num_epoch=<int>`

    Specify number of epochs to train. Default to 16.

### To Test:

Run `test_slim_model.py`

Options:

`--valid_dir`

    Path to test files.
    
`--ckpt_dir`

    Path to saved model.
    
### To Generate Frozen Graph:

Run `freezer.py`

Options:

`--ckpt_dir`
    Path to checkpoint files.
