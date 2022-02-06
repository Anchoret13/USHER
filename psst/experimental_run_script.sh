
RANGE=1000

env_name='Asteroids'
logfile="logging/$env_name.txt"
echo "" > $logfile
args='--replay-k=8 --entropy-regularization=0.05 --n-epochs=5 --n-test-rollouts=10 --n-cycles=100'
# command="mpirun -np 6 python -u train_HER_mod.py --env-name=$env_name $args"
# command="mpirun -np 6 python -u train_HER.py --env-name=$env_name $args"
command="python -u train_HER.py --env-name=$env_name $args"
#mpirun -np 6 python -u train_HER.py --env-name=FetchSlide-v1 --replay-k=8 --entropy-regularization=0.05  --two-goal --apply-ratio
# $command --seed=$(($RANDOM % $RANGE))
# args='--n-epochs=2 --n-test-rollouts=50 --n-cycles=10'
for noise in 0 .1 .25 .5 1.0
do 
	echo -e "\nrunning $env_name, $noise noise, 2-goal ratio" >> $logfile
	for i in 0 1 2 3 4
	do 
		echo "running $env_name, $noise noise, 2-goal ratio"
		$command --noise=$noise --seed=$(($RANDOM % $RANGE)) --two-goal --apply-ratio
	done
	echo -e "\nrunning $env_name, $noise noise, 2-goal" >> $logfile
	for i in 0 1 2 3 4
	do 
		echo "running $env_name, $noise noise, 2-goal"
		$command --noise=$noise --seed=$(($RANDOM % $RANGE)) --two-goal
	done
	echo -e "\nrunning $env_name, $noise noise, 1-goal" >> $logfile
	for i in 0 1 2 3 4
	do 
		echo "running $env_name, $noise noise, 1-goal"
		$command --noise=$noise --seed=$(($RANDOM % $RANGE))
	done
done
