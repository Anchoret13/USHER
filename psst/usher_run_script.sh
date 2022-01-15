

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

env_name='FetchSlide-v1'
args='--n-epochs=20 --n-test-rollouts=100 --n-cycles=150'
command="mpirun -np 3 python -u train_HER.py --env-name=$env_name $args"
for noise in 0 .03 .1 .25 .5
do 
	echo "running $noise noise, 1-goal"
	$command --noise=$noise
	echo "running $noise noise, 2-goal"
	$command --noise=$noise --two-goal
done