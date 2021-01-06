# PM2.5 Research
The research is aim to predict PM2.5 

## Environment
* python3.6.9
* pytorch1.5.0

## Steps
1. Preprocess
```sh
cd dataset
python preprocess.py
```

2. Training
```sh
cd / # the root path of this project
python train.py --no=<the id of training>
```

3. Postprocess
```sh
python test.py --no=<the id of training>
python visual.py --no=<the id of training>
```
