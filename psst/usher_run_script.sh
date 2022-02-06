

# echo "running 0 noise, 1-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=0 
# echo "running 0 noise, 2-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=0 --two-goal
# echo "running .1 noise, 1-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=.1 --n-epochs=20
# echo "running .1 noise, 2-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=.1 --n-epochs=20 --two-goal
# echo "running .25 noise, 1-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=.25
# echo "running .25 noise, 2-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=.25 --two-goal
# echo "running .5 noise, 1-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=.5
# echo "running .5 noise, 2-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=.5 --two-goal

# mpirun -np 6 python -u train_HER_mod.py --env-name='Limited-Range-Based-Navigation-2d-Map4-Goal0-v0'

# env_name='Limited-Range-Based-Navigation-2d-Map4-Goal0-v0'
# args='--n-epochs=10 --n-test-rollouts=100 --n-cycles=150'
# for noise in 0 .01 .03 .1 .25 .5
# do 
# 	echo "running $noise noise, 1-goal"
# 	python -u train_HER_mod.py --env-name=$env_name --noise=$noise $args 
# 	echo "running $noise noise, 2-goal"
# 	python -u train_HER_mod.py --env-name=$env_name --noise=$noise $args --two-goal
# done
RANGE=1000
# env_name='Gridworld'
# logfile="logging/$env_name.txt"
# echo "" > $logfile
# # args='--n-epochs=200 --n-test-rollouts=50 --n-cycles=100 --entropy-regularization=0.01 --gamma=.8 --replay-k=30'
# args='--gamma=.8 --replay-k=30'
# args='--gamma=.8 --replay-k=30'
# args=' '
# command="mpirun -np 6 python -u train_HER_mod.py --env-name=$env_name $args"
# command="python train_HER_mod.py --env-name=$env_name $args"
# command="python train_HER_mod.py --env-name=Gridworld --gamma=.75 --replay-k=50"
# command="python train_HER_mod.py --env-name=Gridworld --gamma=.75 --replay-k=16 --n-epochs=10" # --entropy-regularization=0.05"
# command="python train_HER_mod.py --env-name=Gridworld --gamma=.75 --replay-k=8 --n-epochs=50 --entropy-regularization=0.1"
# command="python train_HER_mod.py --env-name=Gridworld --gamma=.75 --replay-k=6 --n-epochs=10 --entropy-regularization=0.1"
# command="python train_HER_mod.py --env-name=Gridworld --gamma=.775 --replay-k=8 --n-epochs=50 --entropy-regularization=0.1"
# command="python train_HER_mod.py --env-name=Gridworld --gamma=.775 --replay-k=8 --n-epochs=20 --entropy-regularization=0.05"
# # command="mpirun -np 6 python -u train_HER.py --env-name=$env_name $args"
# # run_command="$command --noise=0" 
# # echo "$run_command"
# # $run_command
# # echo "$run_command --two-goal"
# # $run_command --two-goal
# echo -e "\nrunning $env_name, $noise noise, 2-goal, with (ad-hoc, not correct) ratio" >> $logfile
# echo "TWO-GOAL WITH RATIO"
# for i in 0 1 2 3 4 5 
# do 
# 	echo "running $env_name, 2-goal"
# 	$command --seed=$(($RANDOM % $RANGE)) --two-goal --apply-ratio
# 	# echo "command: $run_command"
# 	# $run_command
# done
# echo -e "\nrunning $env_name, $noise noise, 2-goal" >> $logfile
# echo "TWO-GOAL"
# for i in 0 1 2 3 4 5 
# do 
# 	echo "running $env_name, 2-goal"
# 	$command --seed=$(($RANDOM % $RANGE)) --two-goal
# 	# echo "command: $run_command"
# 	# $run_command
# done
# echo -e "\nrunning $env_name, $noise noise, 1-goal" >> $logfile
# echo "ONE-GOAL"
# for i in 0 1 2 3 4 5 
# do 
# 	echo "running $env_name, 1-goal"
# 	$command --seed=$(($RANDOM % $RANGE))
# 	# run_command="$command"
# 	# echo "command: $run_command"
# 	# $run_command
# done


env_name='FetchSlide-v1'
logfile="logging/$env_name.txt"
echo "" > $logfile
args='--replay-k=8 --entropy-regularization=0.05 --n-epochs=50 --n-test-rollouts=50 --n-cycles=100'
# command="mpirun -np 6 python -u train_HER_mod.py --env-name=$env_name $args"
command="mpirun -np 6 python -u train_HER.py --env-name=$env_name $args"
#mpirun -np 6 python -u train_HER.py --env-name=FetchSlide-v1 --replay-k=8 --entropy-regularization=0.05  --two-goal --apply-ratio
# $command --seed=$(($RANDOM % $RANGE))
# args='--n-epochs=2 --n-test-rollouts=50 --n-cycles=10'
for noise in 0 .25 .5 1.0
do 
	# echo -e "\nrunning $env_name, $noise noise, 2-goal ratio" >> $logfile
	# for i in 0 1 2 3 4
	# do 
	# 	echo "running $env_name, $noise noise, 2-goal ratio"
	# 	$command --noise=$noise --seed=$(($RANDOM % $RANGE)) --two-goal --apply-ratio
	# done
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

