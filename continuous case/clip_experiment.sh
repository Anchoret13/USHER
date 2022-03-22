N=6
RANGE=1000

env_name="Asteroids"
# agent="train_HER.py"
agent="train_HER_mod.py"
# offset=".01"
k=8
offset=".01"
args='--entropy-regularization=0.0001 --n-test-rollouts=50 --n-cycles=500 --n-batches=4'
# for env_name in "Gridworld" "RandomGridworld" "AsteroidsGridworld" "RandomAsteroidsGridworld" "CarGridworld" "RandomCarGridworld"
 # "RandomGridworld" "RandomBlockyGridworld"
# for env_name in "RandomGridworld"  "RandomBlockyGridworld"  "AsteroidsRandomGridworld" "AsteroidsRandomBlockyGridworld" #"CarRandomGridworld"
# for env_name in "RandomBlockyGridworld"  "AsteroidsRandomGridworld" "AsteroidsRandomBlockyGridworld" "CarRandomGridworld" "CarRandomBlockyGridworld" 
for env_name in  "AltRandomGridworld"
# for env_name in "AsteroidsRandomGridworld" "AsteroidsRandomBlockyGridworld" #"CarRandomGridworld"
# for env_name in "AsteroidsRandomGridworld" "AsteroidsRandomBlockyGridworld" "CarRandomGridworld" "CarRandomBlockyGridworld" 
# for env_name in "TwoDoorGridworld"
do
	logfile="logging/$env_name.txt"
	# echo "" > $logfile
	epochs=10
	gamma=.9
	command="mpirun -np 1 python -u $agent --env-name=$env_name $args --n-epochs=$epochs --ratio-offset=$offset  --gamma=$gamma --replay-k=$k"
	
	for clip in 0 0.1 0.2 0.3 0.4 0.6 1.0 2.0
	do 
		echo -e "\nrunning $env_name, $clip ratio-clip, 2-goal ratio with $offset offset" >> $logfile
		echo "command=$command --seed=$(($RANDOM % $RANGE)) --two-goal --apply-ratio --ratio-clip=$clip"
		for i in 0 1 2 3 4 5 6 
		do 
			( echo "running $env_name, $noise noise, 2-goal ratio with $offset offset";
			$command --seed=$(($RANDOM % $RANGE)) --two-goal --apply-ratio --ratio-clip=$clip) &
		done
		wait
	done
done