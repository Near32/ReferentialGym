#!/bin/bash
# $1==SEED_BASIS  
# $2==ARCH( /BN+CNN3x3)
# $3==NBR_DISTRACTORS (768 train + 256 test)
# $4==BATCH_SIZE (32/192/384/576/768)

python train.py --seed $(($1+0)) \
--arch $2 \
--epoch 10000 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $3 \
--max_sentence_length 20 --vocab_size 100 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-1-S16-X-1-S16-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $4 &
#--shared_architecture \
#--fast \


python train.py --seed $(($1+10)) \
--arch $2 \
--epoch 10000 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $3 \
--max_sentence_length 20 --vocab_size 100 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-1-S16-X-1-S16-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $4 &
#--shared_architecture \
#--fast \


python train.py --seed $(($1+20)) \
--arch $2 \
--epoch 10000 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $3 \
--max_sentence_length 20 --vocab_size 100 \
--dataset dSprites \
--train_test_split_strategy 2combinatorial2-Y-1-S16-X-1-S16-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $4 &
#--shared_architecture \
#--fast \


python train.py --seed $(($1+30)) \
--arch $2 \
--epoch 10000 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $3 \
--max_sentence_length 20 --vocab_size 100 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-1-S16-X-1-S16-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $4 &
#--shared_architecture \
#--fast \


python train.py --seed $(($1+40)) \
--arch $2 \
--epoch 10000 \
--distractor_sampling uniform \
--nbr_train_distractors $3 --nbr_test_distractors $3 \
--max_sentence_length 20 --vocab_size 100 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-1-S16-X-1-S16-Orientation-40-N-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $4 &
#--shared_architecture \
#--fast \
