#!/bin/bash
# $1==SEED_BASIS  
## $2==ARCH( /BN+CNN3x3)
## $3==NBR_DISTRACTORS_TRAIN (48 train + 16 test)
## $4==NBR_DISTRACTORS_TEST (48 train + 16 test)
#$2 $5==MAX_SENTENCE_LENGTH1 (4/20)
#$3 $6==VOCAB_SIZE1 (9,100)
## 8 $7==BATCH_SIZE (2/12/24/36/48)

python train.py \
--parent_folder EffectCapacity/Spectrum/ \
--seed $(($1+0)) \
--arch BN+CNN3x3 \
--epoch 1875 \
--distractor_sampling uniform \
--nbr_train_distractors 47 --nbr_test_distractors 63 \
--max_sentence_length $2 --vocab_size $3 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size 8 &
#--shared_architecture \
#--fast \


python train.py \
--parent_folder EffectCapacity/Spectrum/ \
--seed $(($1+10)) \
--arch BN+CNN3x3 \
--epoch 1875 \
--distractor_sampling uniform \
--nbr_train_distractors 47 --nbr_test_distractors 63 \
--max_sentence_length $2 --vocab_size $3 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size 8 &
#--shared_architecture \
#--fast \

python train.py \
--parent_folder EffectCapacity/Spectrum/ \
--seed $(($1+20)) \
--arch BN+CNN3x3 \
--epoch 1875 \
--distractor_sampling uniform \
--nbr_train_distractors 47 --nbr_test_distractors 63 \
--max_sentence_length $2 --vocab_size $3 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size 8 
#--shared_architecture \
#--fast \