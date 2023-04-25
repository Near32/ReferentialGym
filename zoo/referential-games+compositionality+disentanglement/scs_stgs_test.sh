WANDB_CACHE_DIR=./wandb_cache/ python -m ipdb -c c train_wandb.py \
--add_descriptive_test=False --add_discriminative_test=False \
--agent_loss_type=Impatient+Hinge --agent_nbr_latent_dim=32 \
--arch=Dummy --baseline_only=False \
--lr=0.001 --weight_decay=1.0e-3 \
--with_logits_mdl_principle=True \
--logits_mdl_principle_factor=1.0e-3 --logits_mdl_principle_accuracy_threshold=70 \
--nbr_experience_repetition=1 --batch_size=128 \
--dataset=SCS --dataset_length=0 \
--scs_nbr_latents=4 --scs_min_nbr_values_per_latent=4 \
--scs_max_nbr_values_per_latent=4 --scs_nbr_object_centric_samples=4 \
--nb_3dshapespybullet_colors=10 --nb_3dshapespybullet_samples=10 \
--nb_3dshapespybullet_shapes=10 --nb_3dshapespybullet_train_colors=5 \
--nbr_discriminative_test_distractors=7 --nbr_distractors=0 \
--descriptive=True --descriptive_ratio=0 \
--dis_metric_resampling=True --distractor_sampling=uniform \
--dropout_prob=0 --emb_dropout_prob=0.0 \
--egocentric=False --egocentric_tr_degrees=15 --egocentric_tr_xy=0.125 \
--epoch=16001 --graphtype=straight_through_gumbel_softmax \
--max_sentence_length=10 --vocab_size=20 \
--metric_active_factors_only=True --metric_batch_size=64 \
--dis_metric_epoch_period=201 --metric_epoch_period=100 \
--metric_resampling=True --mini_batch_size=64 \
--nbr_eval_points=400 --nbr_train_points=500 \
--object_centric=True --obverter_nbr_games_per_round=32 \
--obverter_use_decision_head=False --obverter_learn_not_target_logit=True \
--obverter_nbr_head_outputs=2 --use_obverter_sampling=False \
--obverter_sampling_round_alternation_only=False --obverter_sampling_repeat_experiences=False \
--obverter_threshold_to_stop_message_generation=0.95 --obverter_use_residual_connections=False \
--parallel_TS_worker=32 --parent_folder=./scs_stgs_test \
--resizeDim=None --seed=11 --shared_architecture=True \
--symbol_embedding_size=64 --symbol_processing_nbr_hidden_units=256 \
--synthetic_progression_end=10 --train_test_split_strategy=divider-1-offset-0 \
--use_cuda=True --vae_factor_gamma=0 --vae_gaussian=False --vae_lambda=0 --vae_nbr_latent_dim=32 \
--visual_context_consistent_obverter=False --use_utterance_conditioned_threshold=False \
--with_BN_in_obverter_decision_head=False --with_DP_in_obverter_decision_head=False \
--with_baseline=False --with_color_jitter_augmentation=False \
--with_gaussian_blur_augmentation=False \
--with_classification_test=False --classification_test_nbr_class=10 \
--classification_test_loss_lambda=1.0 --with_attached_classification_heads=False \
--use_aitao=False --use_priority=False \
--aitao_max_similarity_ratio=10.0 \
--aitao_target_unique_prod_ratio=100.0 \
--aitao_language_similarity_sampling_epoch_period=1 \
--object_centric_version=2 --descriptive_version=2 

#--synthetic_progression_end=10 --train_test_split_strategy=combinatorial2-FloorHue-2-S2-WallHue-2-S2-ObjectHue-2-S2-Scale-8-N-Shape-1-N-Orientation-3-N \
