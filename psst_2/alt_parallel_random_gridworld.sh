N=6
RANGE=1000

env_name="Asteroids"
# agent="train_HER.py"
agent="train_HER_mod.py"
# offset=".01"
k=8
offset=.1 #".02"
clip=0.3
# args='--entropy-regularization=0.001 --n-test-rollouts=50 --n-cycles=500 --n-batches=4 --batch-size=1000'
args='--entropy-regularization=0.01 --n-test-rollouts=50 --n-cycles=500 --n-batches=4 --batch-size=1000 --polyak=.99'
# args='--entropy-regularization=0.001 --n-test-rollouts=50'
# for env_name in "Gridworld" "RandomGridworld" "AsteroidsGridworld" "RandomAsteroidsGridworld" "CarGridworld" "RandomCarGridworld"
 # "RandomGridworld" "RandomBlockyGridworld"
# for env_name in "RandomGridworld"  "RandomBlockyGridworld"  "AsteroidsRandomGridworld" "AsteroidsRandomBlockyGridworld" #"CarRandomGridworld"
# for env_name in "RandomBlockyGridworld"  "AsteroidsRandomGridworld" "AsteroidsRandomBlockyGridworld" "CarRandomGridworld" "CarRandomBlockyGridworld" 
# for env_name in  "AsteroidsRandomGridworld" #"StandardCarRandomGridworld" 
for env_name in  "StandardCarRandomGridworld"  # "TorusFreeze6"  
# for env_name in "AsteroidsRandomGridworld" "AsteroidsRandomBlockyGridworld" #"CarRandomGridworld"
# for env_name in "AsteroidsRandomGridworld" "AsteroidsRandomBlockyGridworld" "CarRandomGridworld" "CarRandomBlockyGridworld" 
# for env_name in "TwoDoorGridworld"

do
	logfile="logging/$env_name.txt"
	echo "" > $logfile
	epochs=20
	gamma=.9
	alt_gamma=.9
	# if [[ $env_name == "RandomGridworld" ]]; then
	# 	gamma=$alt_gamma
	# 	epochs=10
	# fi
	# if [[ $env_name == "RandomBlockyGridworld" ]]; then
	# 	gamma=$alt_gamma
	# 	epochs=5
	# fi
	# if [[ $env_name == "AsteroidsRandomGridworld" ]]; then
	# 	epochs=50
	# fi
	# if [[ $env_name == "AsteroidsRandomBlockyGridworld" ]]; then
	# 	epochs=50
	# fi
	# if [[ $env_name  == "RandomAsteroidsGridworld" ]]; then
	# 	gamma=$alt_gamma
	# fi
	# if [[ $env_name  == "RandomCarGridworld" ]]; then
	# 	gamma=$alt_gamma
	# fi
	command="mpirun -np 1 python -u $agent --env-name=$env_name $args --n-epochs=$epochs --ratio-offset=$offset  --gamma=$gamma --replay-k=$k --ratio-clip=$clip"
	echo "command=$command"
	for noise in 0 
	do 
		echo -e "\nrunning $env_name, $noise noise, 2-goal ratio with $offset offset" >> $logfile
		for i in 0  
		do 
			for i in 0 1 2 3 4 5 6 
			do 
				( echo "running $env_name, $noise noise, 2-goal ratio with $offset offset";
				$command --action-noise=$noise --seed=$(($RANDOM % $RANGE)) --two-goal --apply-ratio) &
			done
			wait
		done
		echo -e "\nrunning $env_name, $noise noise, 2-goal" >> $logfile
		# for i in 0 1
		# do 
		# 	for i in 0 1 2 3 4 5 6
		# 	do 
		# 		(echo "running $env_name, $noise noise, 2-goal"; 
		# 		$command --action-noise=$noise --seed=$(($RANDOM % $RANGE)) --two-goal ) &
		# 	done
		# 	wait
		# done
		echo -e "\nrunning $env_name, $noise noise, 1-goal" >> $logfile
		for i in 0 
		do 
			for i in 0 1 2 3 4 5 6
			do 
				(echo "running $env_name, $noise noise, 1-goal"; 
				$command --action-noise=$noise --seed=$(($RANDOM % $RANGE))) &
			done
			wait
		done
		echo -e "\nrunning $env_name, $noise noise, Q-learning" >> $logfile
		# for i in 0 1
		# do 
		# 	for i in 0 1 2 3 4 5 6
		# 	do 
		# 		(echo "running $env_name, $noise noise, Q-learning"; 
		# 		$command --action-noise=$noise --seed=$(($RANDOM % $RANGE)) --replay-k=0) &
		# 	done
		# 	wait
		# done
	done
done