#!/bin/bash
# $1==SEED_BASIS  
# $2==MAX_SENTENCE_LENGTH (10/20)
# $3==VOCAB_SIZE1 (10,100)
# $4==BATCH_SIZE (2/12/24/36/48)
# $5==VAE_GAMMA_FACTOR (0.0--100.0)
# $6==NBR_DISTRACTORRS (0--)
# $7==LOSS_TYPE ("NLL"/"Hinge")
# $8==GRAPHTYPE ("straight_through_gumbel_softmax"/"obverter")
# $9==DESCRIPTIVE_RATIO (0.0--)
# $10==OBJECT_CENTRIC (""/"--object_centric")
# $11==SHARED_ARCH (""/"--shared_architecture")
# $12==SAMPLING/BASELINE (""/"--baseline_only"/"--obverter_sampling_round_alternation_only")
# $13==DESCRIPTIVE (""/"--descriptive")
# $14==CCO (""/"--context_consistent_obverter")
# $15==DecisionHeadBN (""/"--with_BN_in_obverter_decision_head")

python -m ipdb -c c train.py \
--parent_folder ./PoorlyDiverseStimuli/August4th \
--use_cuda --seed $(($1+0)) \
--obverter_nbr_games_per_round 2 --obverter_threshold_to_stop_message_generation 0.0 \
--batch_size $4 --mini_batch_size $4 --vae_lambda 0.0 \
--resizeDim 64 --arch BN+BetaVAEEncoderOnly3x3 --emb_dropout_prob 0.0 --dropout_prob 0.0 \
--max_sentence_length $2 --vocab_size $3 \
--epoch 4001 --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 \
--nbr_train_distractors $6 --nbr_test_distractors $6 \
--obverter_use_decision_head --obverter_nbr_head_outputs 2 \
--agent_loss_type $7 --graphtype $8 \
--metric_epoch_period 20 --nbr_train_points 150 --nbr_eval_points 100 --metric_batch_size 16 \
--dis_metric_resampling --metric_resampling --metric_active_factors_only \
--lr 6e-4 --agent_nbr_latent_dim 32 --vae_nbr_latent_dim 32 \
--descriptive_ratio $9 \
${10} ${11} ${12} ${13} ${14} ${15} \
--synthetic_progression_end 4000 \
--nb_3dshapespybullet_shapes 5 --nb_3dshapespybullet_colors 5 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 3 \
--dataset 3DShapesPyBullet \
--train_test_split_strategy combinatorial2-Y-16-S1-X-16-S1-Orientation-4-N-Scale-2-S1-Shape-1-N & 
#--descriptive 
#--vae_factor_gamma $5 \
#--with_baseline \


python -m ipdb -c c train.py \
--parent_folder ./PoorlyDiverseStimuli/August4th \
--use_cuda --seed $(($1+10)) \
--obverter_nbr_games_per_round 2 --obverter_threshold_to_stop_message_generation 0.0 \
--batch_size $4 --mini_batch_size $4 --vae_lambda 0.0 \
--resizeDim 64 --arch BN+BetaVAEEncoderOnly3x3 --emb_dropout_prob 0.0 --dropout_prob 0.0 \
--max_sentence_length $2 --vocab_size $3 \
--epoch 4001 --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 \
--nbr_train_distractors $6 --nbr_test_distractors $6 \
--obverter_use_decision_head --obverter_nbr_head_outputs 2 \
--agent_loss_type $7 --graphtype $8 \
--metric_epoch_period 20 --nbr_train_points 150 --nbr_eval_points 100 --metric_batch_size 16 \
--dis_metric_resampling --metric_resampling --metric_active_factors_only \
--lr 6e-4 --agent_nbr_latent_dim 32 --vae_nbr_latent_dim 32 \
--descriptive_ratio $9 \
${10} ${11} ${12} ${13} ${14} ${15} \
--synthetic_progression_end 4000 \
--nb_3dshapespybullet_shapes 5 --nb_3dshapespybullet_colors 5 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 3 \
--dataset 3DShapesPyBullet \
--train_test_split_strategy combinatorial2-Y-16-S1-X-16-S1-Orientation-4-N-Scale-2-S1-Shape-1-N &
#--descriptive 
#--vae_factor_gamma $5 \
#--with_baseline \

python -m ipdb -c c train.py \
--parent_folder ./PoorlyDiverseStimuli/August4th \
--use_cuda --seed $(($1+20)) \
--obverter_nbr_games_per_round 2 --obverter_threshold_to_stop_message_generation 0.0 \
--batch_size $4 --mini_batch_size $4 --vae_lambda 0.0 \
--resizeDim 64 --arch BN+BetaVAEEncoderOnly3x3 --emb_dropout_prob 0.0 --dropout_prob 0.0 \
--max_sentence_length $2 --vocab_size $3 \
--epoch 4001 --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 \
--nbr_train_distractors $6 --nbr_test_distractors $6 \
--obverter_use_decision_head --obverter_nbr_head_outputs 2 \
--agent_loss_type $7 --graphtype $8 \
--metric_epoch_period 20 --nbr_train_points 150 --nbr_eval_points 100 --metric_batch_size 16 \
--dis_metric_resampling --metric_resampling --metric_active_factors_only \
--lr 6e-4 --agent_nbr_latent_dim 32 --vae_nbr_latent_dim 32 \
--descriptive_ratio $9 \
${10} ${11} ${12} ${13} ${14} ${15} \
--synthetic_progression_end 4000 \
--nb_3dshapespybullet_shapes 5 --nb_3dshapespybullet_colors 5 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 3 \
--dataset 3DShapesPyBullet \
--train_test_split_strategy combinatorial2-Y-16-S1-X-16-S1-Orientation-4-N-Scale-2-S1-Shape-1-N &
#--descriptive 
#--vae_factor_gamma $5 \
#--with_baseline \

python -m ipdb -c c train.py \
--parent_folder ./PoorlyDiverseStimuli/August4th \
--use_cuda --seed $(($1+30)) \
--obverter_nbr_games_per_round 2 --obverter_threshold_to_stop_message_generation 0.0 \
--batch_size $4 --mini_batch_size $4 --vae_lambda 0.0 \
--resizeDim 64 --arch BN+BetaVAEEncoderOnly3x3 --emb_dropout_prob 0.0 --dropout_prob 0.0 \
--max_sentence_length $2 --vocab_size $3 \
--epoch 4001 --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 \
--nbr_train_distractors $6 --nbr_test_distractors $6 \
--obverter_use_decision_head --obverter_nbr_head_outputs 2 \
--agent_loss_type $7 --graphtype $8 \
--metric_epoch_period 20 --nbr_train_points 150 --nbr_eval_points 100 --metric_batch_size 16 \
--dis_metric_resampling --metric_resampling --metric_active_factors_only \
--lr 6e-4 --agent_nbr_latent_dim 32 --vae_nbr_latent_dim 32 \
--descriptive_ratio $9 \
${10} ${11} ${12} ${13} ${14} ${15} \
--synthetic_progression_end 4000 \
--nb_3dshapespybullet_shapes 5 --nb_3dshapespybullet_colors 5 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 3 \
--dataset 3DShapesPyBullet \
--train_test_split_strategy combinatorial2-Y-16-S1-X-16-S1-Orientation-4-N-Scale-2-S1-Shape-1-N & 
#--descriptive 
#--vae_factor_gamma $5 \
#--with_baseline \

python -m ipdb -c c train.py \
--parent_folder ./PoorlyDiverseStimuli/August4th \
--use_cuda --seed $(($1+40)) \
--obverter_nbr_games_per_round 2 --obverter_threshold_to_stop_message_generation 0.0 \
--batch_size $4 --mini_batch_size $4 --vae_lambda 0.0 \
--resizeDim 64 --arch BN+BetaVAEEncoderOnly3x3 --emb_dropout_prob 0.0 --dropout_prob 0.0 \
--max_sentence_length $2 --vocab_size $3 \
--epoch 4001 --symbol_processing_nbr_hidden_units 64 --symbol_embedding_size 64 \
--nbr_train_distractors $6 --nbr_test_distractors $6 \
--obverter_use_decision_head --obverter_nbr_head_outputs 2 \
--agent_loss_type $7 --graphtype $8 \
--metric_epoch_period 20 --nbr_train_points 150 --nbr_eval_points 100 --metric_batch_size 16 \
--dis_metric_resampling --metric_resampling --metric_active_factors_only \
--lr 6e-4 --agent_nbr_latent_dim 32 --vae_nbr_latent_dim 32 \
--descriptive_ratio $9 \
${10} ${11} ${12} ${13} ${14} ${15} \
--synthetic_progression_end 4000 \
--nb_3dshapespybullet_shapes 5 --nb_3dshapespybullet_colors 5 --nb_3dshapespybullet_samples 10 --nb_3dshapespybullet_train_colors 3 \
--dataset 3DShapesPyBullet \
--train_test_split_strategy combinatorial2-Y-16-S1-X-16-S1-Orientation-4-N-Scale-2-S1-Shape-1-N
#--descriptive 
#--vae_factor_gamma $5 \
#--with_baseline \

