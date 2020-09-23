python -m ipdb -c c train_stgs.py \
--parent_folder ./debug/Sept22/STGS/\
--use_cuda --seed 0 \
--emb_dropout_prob 0.0 --dropout_prob 0.0 --use_sentences_one_hot_vectors \
--batch_size 128 --mini_batch_size 256 --resizeDim 128 --arch ShortBaselineCNN \
--descriptive --descriptive_ratio 0.25 \
--max_sentence_length 20 --vocab_size 100 --epoch 10000 \
--symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 \
--object_centric --nbr_train_distractors 32 --nbr_test_distractors 32 \
--agent_loss_type Hinge \
--metric_fast --metric_epoch_period 20

#--egocentric

#--emb_dropout_prob 0.5 --dropout_prob 0.0 --use_sentences_one_hot_vectors \

#--resizeDim 32 --arch BN+3xCNN3x3
#--resizeDim 64 --arch BN+BaselineCNN
