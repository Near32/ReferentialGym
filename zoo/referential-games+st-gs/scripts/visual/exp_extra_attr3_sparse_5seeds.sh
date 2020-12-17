#!/bin/bash
# $1==SEED_BASIS  
# $2==ARCH( /BN+CNN3x3)
# $3==N_EPOCHS
# $4==NBR_DISTRACTORS_TRAIN (48 train + 16 test)
# $5==NBR_DISTRACTORS_TEST (48 train + 16 test)
# $6==MAX_SENTENCE_LENGTH (3/20)
# $7==VOCAB_SIZE (9/100)
# $8==BATCH_SIZE (8/64/128/192/256)
# $9==PARENT_FOLDER

python train.py --seed $(($1+0)) \
--arch $2 \
--epoch $3 \
--distractor_sampling uniform \
--nbr_train_distractors $4 --nbr_test_distractors $5 \
--max_sentence_length $6 --vocab_size $7 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $8 \
--parent_folder $9 &
#--shared_architecture \
#--fast \


python train.py --seed $(($1+10)) \
--arch $2 \
--epoch $3 \
--distractor_sampling uniform \
--nbr_train_distractors $4 --nbr_test_distractors $5 \
--max_sentence_length $6 --vocab_size $7 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $8 \
--parent_folder $9 &
#--shared_architecture \
#--fast \


python train.py --seed $(($1+20)) \
--arch $2 \
--epoch $3 \
--distractor_sampling uniform \
--nbr_train_distractors $4 --nbr_test_distractors $5 \
--max_sentence_length $6 --vocab_size $7 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $8 \
--parent_folder $9 &
#--shared_architecture \
#--fast \


python train.py --seed $(($1+30)) \
--arch $2 \
--epoch $3 \
--distractor_sampling uniform \
--nbr_train_distractors $4 --nbr_test_distractors $5 \
--max_sentence_length $6 --vocab_size $7 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $8 \
--parent_folder $9 &
#--shared_architecture \
#--fast \


python train.py --seed $(($1+40)) \
--arch $2 \
--epoch $3 \
--distractor_sampling uniform \
--nbr_train_distractors $4 --nbr_test_distractors $5 \
--max_sentence_length $6 --vocab_size $7 \
--dataset dSprites \
--train_test_split_strategy combinatorial2-Y-4-S4-X-4-S4-Orientation-5-S4-Scale-6-N-Shape-3-N \
--use_cuda \
--batch_size $8 \
--parent_folder $9
#--shared_architecture \
#--fast \
