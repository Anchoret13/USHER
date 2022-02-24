ACTION_DISCRETISATION = 5
ACTION_NOISE = 0.1
BACKTRACK_COEFF = 0.8
BACKTRACK_ITERS = 10
CONJUGATE_GRADIENT_ITERS = 20
DAMPING_COEFF = 0.1
DISCOUNT = 0.99
EPSILON = 0.05
ENTROPY_WEIGHT = 0.2
HIDDEN_SIZE = 64
KL_LIMIT = 0.0025
# KL_LIMIT = 0.01
LEARNING_RATE = 0.001
# MAX_STEPS = 2*10**6
MAX_STEPS = 3*10**5
# MAX_STEPS = 5*10**1
ON_POLICY_BATCH_SIZE = 2048
OFF_POLICY_BATCH_SIZE = 128
# OFF_POLICY_BATCH_SIZE = 512
POLICY_DELAY = 2
POLYAK_FACTOR = 0.995
PPO_CLIP_RATIO = 0.2
PPO_EPOCHS = 15
REPLAY_SIZE = 100000
TARGET_ACTION_NOISE = 0.2
TARGET_ACTION_NOISE_CLIP = 0.5
TARGET_UPDATE_INTERVAL = 2500
TRACE_DECAY = 0.97
UPDATE_INTERVAL = 10
UPDATE_START = 10000
TEST_INTERVAL = 10000

#------------------------------------------------

# ON_POLICY_BATCH_SIZE =1024
ON_POLICY_BATCH_SIZE =4096
CONJUGATE_GRADIENT_ITERS=10
# DAMPING_COEFF=0.01
DISCOUNT=0.99
# KL_LIMIT=0.02
# LEARNING_RATE=0.0003
VF_ITERS=30

# UPDATE_INTERVAL = 50
# OFF_POLICY_BATCH_SIZE = 512
# LEARNING_RATE = 0.002

#Replicating SAC results
# UPDATE_INTERVAL = 1
# # UPDATE_START = 500
# REPLAY_SIZE = 10**6
# OFF_POLICY_BATCH_SIZE = 256

import torch

MULTIPROCESSING = False
#MULTIPROCESSING = True

#Avoid a bug in torch that throws an error if multiprocessing is used after any call to cuda
if MULTIPROCESSING:
  CUDA = False
else:
  CUDA = torch.cuda.is_available()
CUDA = False
DEVICE=torch.device('cuda' if CUDA else 'cpu')
print("Device: " + str(CUDA))
