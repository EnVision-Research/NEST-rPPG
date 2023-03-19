# A_run
gpu_ids=2
python train.py -g $gpu_ids -t 'VIPL'
python train.py -g $gpu_ids -t 'BUAA'
python train.py -g $gpu_ids -t 'V4V'
python train.py -g $gpu_ids -t 'UBFC'
python train.py -g $gpu_ids -t 'PURE'
