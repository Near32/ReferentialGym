python -m ipdb -c c train_wandb.py \
--project=AITAO-Debug \
--seed=10 \
--add_descriptive_test=True --add_discriminative_test=False \
--agent_loss_type=Hinge --agent_nbr_latent_dim=32 \
--arch=BN+BetaVAEEncoderOnly3x3 \
--baseline_only=False --force_eos=False \
--batch_size=16 --fast=True \
--dataset=3DShapesPyBullet --dataset_length=4096 \
--descriptive=True --descriptive_ratio=0 \
--metric_fast=False \
--dis_metric_epoch_period=2 --dis_metric_resampling=True \
--distractor_sampling=uniform \
--dropout_prob=0 --emb_dropout_prob=0.0 \
--egocentric=False \
--epoch=1001 \
--graphtype=obverter --lr=0.0001 \
--l1_reg_lambda=0.0 --l2_reg_lambda=0.0 \
--max_sentence_length=10 \
--metric_active_factors_only=True \
--metric_batch_size=64 --metric_epoch_period=5 \
--metric_resampling=True --mini_batch_size=64 \
--nb_3dshapespybullet_colors=10 --nb_3dshapespybullet_samples=10 \
--nb_3dshapespybullet_shapes=10 --nb_3dshapespybullet_train_colors=8 \
--nbr_discriminative_test_distractors=7 \
--nbr_distractors=7 \
--nbr_eval_points=500 --nbr_train_points=500 \
--object_centric=True --object_centric_type=hard \
--use_object_centric_curriculum=False \
--object_centric_curriculum_update_epoch_period=4 \
--object_centric_curriculum_accuracy_threshold=50 \
--obverter_learn_not_target_logit=True \
--obverter_nbr_games_per_round=64 --obverter_nbr_head_outputs=2 \
--obverter_sampling_repeat_experiences=False --obverter_sampling_round_alternation_only=True \
--use_obverter_sampling=False --obverter_threshold_to_stop_message_generation=0.75 \
--obverter_use_decision_head=False \
--parallel_TS_worker=32 \
--parent_folder=./PyBullet3DShapes_obverter_aita_test \
--resizeDim=64 \
--shared_architecture=True \
--symbol_embedding_size=64 --symbol_processing_nbr_hidden_units=64 \
--synthetic_progression_end=10 \
--train_test_split_strategy=combinatorial2-FloorHue-2-S2-WallHue-2-S2-ObjectHue-2-S2-Scale-8-N-Shape-1-N-Orientation-3-N \
--use_cuda=True \
--vae_factor_gamma=0 --vae_gaussian=False \
--vae_lambda=0 --vae_nbr_latent_dim=32 \
--visual_context_consistent_obverter=False \
--use_utterance_conditioned_threshold=False \
--vocab_size=40 \
--with_BN_in_obverter_decision_head=False \
--with_DP_in_obverter_decision_head=False \
--with_baseline=False \
--with_color_jitter_augmentation=False \
--with_gaussian_blur_augmentation=True \
--with_classification_test=True --classification_test_nbr_class=15 \
--classification_test_loss_lambda=1.0 --with_attached_classification_heads=False \
--use_aitao=True --use_priority=False \
--aitao_max_similarity_ratio=50.0 --aitao_target_unique_prod_ratio=10.0 \
--aitao_language_similarity_sampling_epoch_period=1 \
--object_centric_version=2 --descriptive_version=1 \
--distractors_sampling_scheme_version=1 \
--distractors_sampling_scheme_with_replacement=True \
--obverter_use_residual_connections=False

# --dataset=3dshapes
