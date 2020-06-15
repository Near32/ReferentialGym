#!/bin/bash
# $1==SEED_BASIS  
# $2==ARCH( /BN+CNN3x3)
# $3==N_EPOCHS (10000)
# $4==MAX_SENTENCE_LENGTH (2/20)
# $5==VOCAB_SIZE (9/100)
# $6==BATCH_SIZE (2/12/24/36/48)
# $7==PARENT_FOLDER

python train_generative.py --seed $(($1+0)) \
--arch $2 \
--epoch $3 \
--distractor_sampling uniform \
--max_sentence_length $4 --vocab_size $5 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda --symbolic --agent_loss_type CE \
--batch_size $6 \
--parent_folder $7 &
#--shared_architecture \
#--fast \


python train_generative.py --seed $(($1+10)) \
--arch $2 \
--epoch $3 \
--distractor_sampling uniform \
--max_sentence_length $4 --vocab_size $5 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda --symbolic --agent_loss_type CE \
--batch_size $6 \
--parent_folder $7 &
#--shared_architecture \
#--fast \


python train_generative.py --seed $(($1+20)) \
--arch $2 \
--epoch $3 \
--distractor_sampling uniform \
--max_sentence_length $4 --vocab_size $5 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda --symbolic --agent_loss_type CE \
--batch_size $6 \
--parent_folder $7 &
#--shared_architecture \
#--fast \


python train_generative.py --seed $(($1+30)) \
--arch $2 \
--epoch $3 \
--distractor_sampling uniform \
--max_sentence_length $4 --vocab_size $5 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda --symbolic --agent_loss_type CE \
--batch_size $6 \
--parent_folder $7 &
#--shared_architecture \
#--fast \


python train_generative.py --seed $(($1+40)) \
--arch $2 \
--epoch $3 \
--distractor_sampling uniform \
--max_sentence_length $4 --vocab_size $5 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda --symbolic --agent_loss_type CE \
--batch_size $6 \
--parent_folder $7
#--shared_architecture \
#--fast \