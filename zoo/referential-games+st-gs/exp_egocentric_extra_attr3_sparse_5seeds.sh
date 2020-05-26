#!/bin/bash
# $1==SEED_BASIS  
# $2==ARCH( /BN+CNN3x3)
# $3==NBR_DISTRACTORS_TRAIN (48 train + 16 test)
# $4==NBR_DISTRACTORS_TEST (48 train + 16 test)
# $5==MAX_SENTENCE_LENGTH (3/20)
# $6==VOCAB_SIZE (9/100)
# $7==BATCH_SIZE (2/12/24/36/48)

python train.py --seed $(($1+0)) \
--arch $2 \
--epoch 1875 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $4 \
--max_sentence_length $5 --vocab_size $6 \
--dataset dSprites --egocentric \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $7 &
#--shared_architecture \
#--fast \


python train.py --seed $(($1+10)) \
--arch $2 \
--epoch 1875 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $4 \
--max_sentence_length $5 --vocab_size $6 \
--dataset dSprites --egocentric \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $7 &
#--shared_architecture \
#--fast \


python train.py --seed $(($1+20)) \
--arch $2 \
--epoch 1875 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $4 \
--max_sentence_length $5 --vocab_size $6 \
--dataset dSprites --egocentric \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $7 &
#--shared_architecture \
#--fast \


python train.py --seed $(($1+30)) \
--arch $2 \
--epoch 1875 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $4 \
--max_sentence_length $5 --vocab_size $6 \
--dataset dSprites --egocentric \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $7 &
#--shared_architecture \
#--fast \


python train.py --seed $(($1+40)) \
--arch $2 \
--epoch 1875 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $4 \
--max_sentence_length $5 --vocab_size $6 \
--dataset dSprites --egocentric \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $7

#--shared_architecture \
#--fast \
