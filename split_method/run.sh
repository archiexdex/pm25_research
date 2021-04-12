echo "train all"
python train.py --no=$1 --model=$2 --method=all
echo "train extreme"
python train.py --no=$1 --model=$2 --method=extreme
echo "train normal"
python train.py --no=$1 --model=$2 --method=normal
echo "train merged"
python train.py --no=$1 --model=$2 --method=merged
