echo "train all"
python train.py --no=$1 --model=$2 --method=all     --target_size=2
echo "train extreme"
python train.py --no=$1 --model=$2 --method=extreme --target_size=2
echo "train normal"
python train.py --no=$1 --model=$2 --method=normal  --target_size=2
echo "train merged"
python train.py --no=$1 --model=$2 --method=merged  --target_size=2
