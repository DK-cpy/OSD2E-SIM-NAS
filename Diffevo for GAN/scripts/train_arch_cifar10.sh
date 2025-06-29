
python fully_train_arch.py \
--gpu_ids 3 \
--num_workers 8 \
--gen_bs 256 \
--dis_bs 128 \
--dataset cifar10 \
--bottom_width 4 \
--img_size 32 \
--max_epoch_G 121 \
--n_critic 5 \
--arch arch_cifar10 \
--draw_arch False \
--genotypes_exp cifar10_D.npy \
--latent_dim 120 \
--gf_dim 256 \
--df_dim 128 \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--val_freq 5 \
--num_eval_imgs 50000 \
--exp_name arch_train_cifar10 \
--data_path /data/datasets/cifar-10 \
--cr 1 \
--genotype_of_G arch_searchG_cifar10_2025_03_08_16_10_29/Model/best_fid_gen.npy \
--use_basemodel_D False
# --genotype_of_G best_gen_0.npy \