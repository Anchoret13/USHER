
for i in 0 1 2 3 4 5 6 
do 
	mpirun -np 1 python -u train_HER_mod.py --env-name=StandardCarRandomGridworld --entropy-regularization=0.01 --n-test-rollouts=50 --n-cycles=500 --n-batches=4 --batch-size=1000 --polyak=.99 --n-epochs=30 --ratio-offset=.02  --gamma=.9 --replay-k=8 --ratio-clip=0.3 --two-goal --apply-ratio &
done
wait
