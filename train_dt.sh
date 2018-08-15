python3 ./train_slim_model.py --save_dir=./model/dt_0815 --no_append --data_dir=./origin-data/road-train-2+valid.v2/train --num_epoch=16
python3 ./train_slim_model.py --resume_dir=./model/dt_0815 --no_append --save_dir ./model/dt_0814-2 --data_dir=./origin-data/contest/train_data --num_epoch=16
