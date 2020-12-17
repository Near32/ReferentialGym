#!/bin/bash
# $1==SEED_BASIS  
# $2==ARCH( /BN+CNN3x3)
# $3==N_EPOCHS
# $4==MAX_SENTENCE_LENGTH (3/20)
# $5==VOCAB_SIZE (9/100)
# $6==BATCH_SIZE (8/64/128/192/256)
# $7==PARENT_FOLDER

python train_generative.py --seed $(($1+0)) \
--arch $2 \
--epoch $3 \
--distractor_sampling uniform \
--max_sentence_length $4 --vocab_size $5 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-2-X-4-2-Orientation-5-2-Scale-6-N-Shape-3-N \
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
--train_test_split_strategy combinatorial2-Y-4-2-X-4-2-Orientation-5-2-Scale-6-N-Shape-3-N \
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
--train_test_split_strategy combinatorial2-Y-4-2-X-4-2-Orientation-5-2-Scale-6-N-Shape-3-N \
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
--train_test_split_strategy combinatorial2-Y-4-2-X-4-2-Orientation-5-2-Scale-6-N-Shape-3-N \
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
--train_test_split_strategy combinatorial2-Y-4-2-X-4-2-Orientation-5-2-Scale-6-N-Shape-3-N \
--use_cuda --symbolic --agent_loss_type CE \
--batch_size $6 \
--parent_folder $7
#--shared_architecture \
#--fast \