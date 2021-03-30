echo "train all"
python train_all.py    --no=$1 
echo "train extreme"
python train_ext.py    --no=$1
echo "train normal"
python train_normal.py --no=$1
echo "train merged"
python train_merged.py --no=$1
