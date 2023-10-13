#CUDA_VISIBLE_DEVICES=0 python -m ipdb -c c train_wandb.py --add_descriptive_test=True --add_discriminative_test=True --agent_loss_type=NLL --agent_nbr_latent_dim=32 --arch=BN+BetaVAEEncoderOnly3x3 --baseline_only=False --batch_size=64 --dataset=3dshapes --dataset_length=2048 --descriptive=True --descriptive_ratio=0 --dis_metric_resampling=True --distractor_sampling=uniform --dropout_prob=0 --egocentric=True --emb_dropout_prob=0 --epoch=301 --graphtype=obverter --lr=0.0001 --max_sentence_length=10 --metric_active_factors_only=True --metric_batch_size=64 --metric_epoch_period=5 --metric_resampling=True --mini_batch_size=64 --nb_3dshapespybullet_colors=10 --nb_3dshapespybullet_samples=10 --nb_3dshapespybullet_shapes=10 --nb_3dshapespybullet_train_colors=5 --nbr_discriminative_test_distractors=7 --nbr_distractors=0 --nbr_eval_points=500 --nbr_train_points=500 --object_centric=False --obverter_nbr_games_per_round=32 --obverter_nbr_head_outputs=2 --obverter_sampling=False --obverter_sampling_round_alternation_only=False --obverter_threshold_to_stop_message_generation=0.75 --obverter_use_decision_head=True --parallel_TS_worker=32 --parent_folder=./3dshapes_stgs_probing_test --resizeDim=64 --seed=11 --shared_architecture=True --symbol_embedding_size=64 --symbol_processing_nbr_hidden_units=128 --synthetic_progression_end=10 --train_test_split_strategy=combinatorial2-FloorHue-2-S2-WallHue-2-S2-ObjectHue-2-S2-Scale-8-N-Shape-1-N-Orientation-3-N --use_cuda=True --vae_factor_gamma=0 --vae_gaussian=False --vae_lambda=0 --vae_nbr_latent_dim=32 --visual_context_consistent_obverter=False --use_utterance_conditioned_threshold=False --vocab_size=20 --with_BN_in_obverter_decision_head=False --with_DP_in_obverter_decision_head=False --with_baseline=False --with_color_jitter_augmentation=True --with_gaussian_blur_augmentation=True --with_classification_test=True --classification_test_nbr_class=15 --classification_test_loss_lambda=10.0 --with_attached_classification_heads=False --use_aitao=True --use_priority=True --aitao_target_unique_prod_ratio=100.0 --aitao_language_similarity_sampling_epoch_period=2 --object_centric_version=2

CUDA_VISIBLE_DEVICES=0 python -m ipdb -c c train_wandb.py \
--add_descriptive_test=True --add_discriminative_test=True \
--agent_loss_type=NLL --agent_nbr_latent_dim=32 \
--arch=BN+BetaVAEEncoderOnly3x3 \
--baseline_only=False \
--batch_size=64 \
--dataset=3dshapes --dataset_length=2048 \
--descriptive=True --descriptive_ratio=0 \
--dis_metric_resampling=True \
--distractor_sampling=uniform \
--dropout_prob=0 --emb_dropout_prob=0 \
--egocentric=True \
--epoch=1001 \
--graphtype=obverter --lr=0.0001 \
--max_sentence_length=10 \
--metric_active_factors_only=True \
--metric_batch_size=64 --metric_epoch_period=5 \
--metric_resampling=True --mini_batch_size=64 \
--nb_3dshapespybullet_colors=10 --nb_3dshapespybullet_samples=10 \
--nb_3dshapespybullet_shapes=10 --nb_3dshapespybullet_train_colors=5 \
--nbr_discriminative_test_distractors=7 \
--nbr_distractors=0 \
--nbr_eval_points=500 --nbr_train_points=500 \
--object_centric=False \
--obverter_nbr_games_per_round=32 --obverter_nbr_head_outputs=2 \
--obverter_sampling=False --obverter_sampling_round_alternation_only=True \
--obverter_threshold_to_stop_message_generation=0.75 \
--obverter_use_decision_head=True \
--parallel_TS_worker=32 \
--parent_folder=./3dshapes_stgs_probing_test \
--resizeDim=64 \
--seed=11 \
--shared_architecture=True \
--symbol_embedding_size=64 --symbol_processing_nbr_hidden_units=64 \
--synthetic_progression_end=10 \
--train_test_split_strategy=combinatorial2-FloorHue-2-S2-WallHue-2-S2-ObjectHue-2-S2-Scale-8-N-Shape-1-N-Orientation-3-N \
--use_cuda=True \
--vae_factor_gamma=0 --vae_gaussian=False \
--vae_lambda=0 --vae_nbr_latent_dim=32 \
--visual_context_consistent_obverter=False \
--use_utterance_conditioned_threshold=False \
--vocab_size=20 \
--with_BN_in_obverter_decision_head=False \
--with_DP_in_obverter_decision_head=False \
--with_baseline=False \
--with_color_jitter_augmentation=True \
--with_gaussian_blur_augmentation=True \
--with_classification_test=True --classification_test_nbr_class=15 \
--classification_test_loss_lambda=10.0 --with_attached_classification_heads=False \
--use_aitao=True --use_priority=False \
--aitao_max_similarity_ratio=75.0 --aitao_target_unique_prod_ratio=10.0 \
--aitao_language_similarity_sampling_epoch_period=3 \
--object_centric_version=2 --descriptive_version=2 \
--obverter_use_residual_connections=False

#CUDA_VISIBLE_DEVICES=4 python -m ipdb -c c train_wandb.py --add_descriptive_test=True --add_discriminative_test=True --agent_loss_type=NLL --agent_nbr_latent_dim=32 --arch=BN+BetaVAEEncoderOnly3x3 --baseline_only=False --batch_size=64 --dataset=3dshapes --dataset_length=2048 --descriptive=True --descriptive_ratio=0 --dis_metric_resampling=True --distractor_sampling=uniform --dropout_prob=0 --egocentric=True --emb_dropout_prob=0 --epoch=301 --graphtype=obverter --lr=0.0001 --max_sentence_length=10 --metric_active_factors_only=True --metric_batch_size=64 --metric_epoch_period=5 --metric_resampling=True --mini_batch_size=64 --nb_3dshapespybullet_colors=10 --nb_3dshapespybullet_samples=10 --nb_3dshapespybullet_shapes=10 --nb_3dshapespybullet_train_colors=5 --nbr_discriminative_test_distractors=7 --nbr_distractors=3 --nbr_eval_points=500 --nbr_train_points=500 --object_centric=False --obverter_nbr_games_per_round=32 --obverter_nbr_head_outputs=2 --obverter_sampling=False --obverter_sampling_round_alternation_only=True --obverter_threshold_to_stop_message_generation=0.75 --obverter_use_decision_head=True --parallel_TS_worker=32 --parent_folder=./3dshapes_obverter_probing_test --resizeDim=64 --seed=11 --shared_architecture=True --symbol_embedding_size=64 --symbol_processing_nbr_hidden_units=128 --synthetic_progression_end=10 --train_test_split_strategy=combinatorial2-FloorHue-2-S2-WallHue-2-S2-ObjectHue-2-S2-Scale-8-N-Shape-1-N-Orientation-3-N --use_cuda=True --vae_factor_gamma=0 --vae_gaussian=False --vae_lambda=0 --vae_nbr_latent_dim=32 --visual_context_consistent_obverter=False --use_utterance_conditioned_threshold=True --vocab_size=20 --with_BN_in_obverter_decision_head=False --with_DP_in_obverter_decision_head=False --with_baseline=False --with_color_jitter_augmentation=True --with_gaussian_blur_augmentation=True --with_classification_test=True --classification_test_loss_lambda=10.0 --with_attached_classification_heads=False --use_aitao=True --aitao_target_unique_prod_ratio=100.0 --aitao_language_similarity_sampling_epoch_period=2

#_UDA_VISIBLE_DEVICES=3 python -m ipdb -c c train_wandb.py --add_descriptive_test=True --add_discriminative_test=True --agent_loss_type=NLL --agent_nbr_latent_dim=32 --arch=BN+BetaVAEEncoderOnly3x3 --baseline_only=False --batch_size=64 --dataset=3dshapes --dataset_length=2048 --descriptive=True --descriptive_ratio=0 --dis_metric_resampling=True --distractor_sampling=uniform --dropout_prob=0 --egocentric=True --emb_dropout_prob=0 --epoch=301 --graphtype=obverter --lr=0.0001 --max_sentence_length=10 --metric_active_factors_only=True --metric_batch_size=64 --metric_epoch_period=2 --metric_resampling=True --mini_batch_size=64 --nb_3dshapespybullet_colors=10 --nb_3dshapespybullet_samples=10 --nb_3dshapespybullet_shapes=10 --nb_3dshapespybullet_train_colors=5 --nbr_discriminative_test_distractors=7 --nbr_distractors=3 --nbr_eval_points=500 --nbr_train_points=500 --object_centric=False --obverter_nbr_games_per_round=2 --obverter_nbr_head_outputs=2 --obverter_sampling=False --obverter_sampling_round_alternation_only=True --obverter_threshold_to_stop_message_generation=0.75 --obverter_use_decision_head=True --parallel_TS_worker=32 --parent_folder=./3dshapes_obverter_probing_test --resizeDim=64 --seed=11 --shared_architecture=True --symbol_embedding_size=64 --symbol_processing_nbr_hidden_units=128 --synthetic_progression_end=10 --train_test_split_strategy=combinatorial2-FloorHue-2-S2-WallHue-2-S2-ObjectHue-2-S2-Scale-8-N-Shape-1-N-Orientation-3-N --use_cuda=True --vae_factor_gamma=0 --vae_gaussian=False --vae_lambda=0 --vae_nbr_latent_dim=32 --visual_context_consistent_obverter=False --vocab_size=20 --with_BN_in_obverter_decision_head=False --with_DP_in_obverter_decision_head=False --with_baseline=False --with_color_jitter_augmentation=True --with_gaussian_blur_augmentation=True --with_classification_test=True --classification_test_loss_lambda=10.0 --with_attached_classification_heads=False --use_aitao=True --aitao_target_unique_prod_ratio=100.0 --aitao_language_similarity_sampling_epoch_period=2

