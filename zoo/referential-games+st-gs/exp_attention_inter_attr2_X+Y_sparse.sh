#!/bin/bash
# $1==SEED_BASIS  
# $2==ARCH( /BN+CNN3x3)
# $3==NBR_DISTRACTORS_TRAIN (48 train + 16 test)
# $4==NBR_DISTRACTORS_TEST (48 train + 16 test)
# $5==MAX_SENTENCE_LENGTH (2/5/20)
# $6==VOCAB_SIZE (9/17/100)
# $7==BATCH_SIZE (2/12/24/36/48)

python train_attention.py --seed $(($1+0)) \
--agent_type AttentionListener \
--arch $2 \
--epoch 10000 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $4 \
--max_sentence_length $5 --vocab_size $6 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-2-X-4-2-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $7 &
#--shared_architecture \
#--fast \


python train_attention.py --seed $(($1+10)) \
--agent_type AttentionListener \
--arch $2 \
--epoch 10000 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $4 \
--max_sentence_length $5 --vocab_size $6 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-2-X-4-2-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $7 &
#--shared_architecture \
#--fast \


python train_attention.py --seed $(($1+20)) \
--agent_type AttentionListener \
--arch $2 \
--epoch 10000 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $4 \
--max_sentence_length $5 --vocab_size $6 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-2-X-4-2-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $7 &
#--shared_architecture \
#--fast \


python train_attention.py --seed $(($1+30)) \
--agent_type AttentionListener \
--arch $2 \
--epoch 10000 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $4 \
--max_sentence_length $5 --vocab_size $6 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-2-X-4-2-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $7 &
#--shared_architecture \
#--fast \


python train_attention.py --seed $(($1+40)) \
--agent_type AttentionListener \
--arch $2 \
--epoch 10000 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $4 \
--max_sentence_length $5 --vocab_size $6 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-2-X-4-2-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $7

#--shared_architecture \
#--fast \
