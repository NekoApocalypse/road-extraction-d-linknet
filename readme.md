# D-Link Net for Road Extraction

## Using Pretrained Res50 Network from Tensorflow-Slim

Implements the network from: D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction

Modified to work with Res50.

## Usage

To train: run 'train_slim_model.py'

To test: run 'test_slim_mmodel.py'

## Additional Notes

1. *To do*: Retrain model on CVPR dataset.

2. *To Do*: Modify data loader to read from contest dataset.

3. **IMPORTANT** Back up checkpoint files in './model' to another directory, and use the backed up files as **resume_path** to continue training. **train_slim_model.py will override existing checkpoints**.

4. Set **resume_path** argument is set in train_slim_model.py and train on contest dataset.