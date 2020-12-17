python -m ipdb -c c train_sqoot.py \
--parent_folder /home/kevin/debugging_RG/RR/debug_from_utterances/August20/debugEgoBlack/ \
--seed 10 --use_cuda --fast --metric_fast \
--nbr_train_distractors 31 --nbr_test_distractors 31 \
--dropout_prob 0.0 --emb_dropout_prob 0.0 \
--vocab_size 100 --max_sentence_length 20 --gumbel_softmax_tau0 0.001 \
--shared_architecture --same_head \
--nb_sqoot_samples 512 --nb_sqoot_objects 5 --nb_sqoot_shapes 36 --nb_train_rhs 18 \
--from_utterances --egocentric --object_centric --egocentric_tr_degrees 6 --egocentric_tr_xy 0.0625 \
--with_baseline

#--fast
# TODO: nb_train_rhs != 18 is not possible because the values cannot be stacked, cf l173 in dual_labeled_dataset
# and also in the reshape of the features: train and test-time reshape do not agree...
# need to implement a conditional  / mutex ...

# TODO: there seems to be an inbalance in the binary relation queries possible answers...
# CUDA_LAUNCH_BLOCKING=1 

#precision_recall_fscore_support(y_true=gtv,y_pred=pv,labels=self.config["labels"],average='macro',sample_weight=sample_weights,zero_division=0)
#precision_recall_fscore_support(y_true=gtv,y_pred=pv,labels=self.config["labels"],average='macro',sample_weight=sample_weights,zero_division=0)