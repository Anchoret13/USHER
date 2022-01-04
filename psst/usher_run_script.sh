# echo "running 0 noise, 1-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=0 
# echo "running 0 noise, 2-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=0 --two-goal
echo "running .1 noise, 1-goal"
mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=.1 --n-epochs=20
echo "running .1 noise, 2-goal"
mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=.1 --n-epochs=20 --two-goal
# echo "running .25 noise, 1-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=.25
# echo "running .25 noise, 2-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=.25 --two-goal
# echo "running .5 noise, 1-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=.5
# echo "running .5 noise, 2-goal"
# mpirun -np 3 python -u train_HER.py --env-name='FetchSlide-v1' --action-noise=.5 --two-goal