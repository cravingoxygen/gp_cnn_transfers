#!/bin/bash
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_cw_l2_conf=0.0_max_iter=20_init_c=10.0_lr=0.5' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_cw_l2_conf=2.0_max_iter=20_init_c=10.0_lr=0.5' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_cw_l2_conf=5.0_max_iter=20_init_c=10.0_lr=0.5' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_pgd_eps=0.07_eps_iter=0.007_nb_iter=10_ord=inf' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_pgd_eps=0.09_eps_iter=0.009_nb_iter=10_ord=inf' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_pgd_eps=0.2_eps_iter=0.02_nb_iter=10_ord=inf' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_pgd_eps=0.4_eps_iter=0.04_nb_iter=10_ord=inf' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_pgd_eps=0.5_eps_iter=0.05_nb_iter=10_ord=inf' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_pgd_eps=1.0_eps_iter=0.1_nb_iter=10_ord=2.0' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_pgd_eps=2.0_eps_iter=0.2_nb_iter=10_ord=2.0' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_pgd_eps=5.0_eps_iter=0.5_nb_iter=10_ord=2.0' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_pgd_eps=10.0_eps_iter=1.0_nb_iter=10_ord=2.0' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_pgd_eps=20.0_eps_iter=2.0_nb_iter=10_ord=2.0' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_fgsm_eps=1.0_norm=2.0' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_fgsm_eps=2.0_norm=2.0' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_fgsm_eps=5.0_norm=2.0' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_fgsm_eps=10.0_norm=2.0' -action='attack'
python3 cnn.py -data_dir='/scratch/etv21/conv_gp_data/MNIST_data/expA2' \
    -output_path='/scratch/etv21/conv_gp_data/expA4' \
    -report='report_transfer_23_09_19.txt' -model='/scratch/etv21/conv_gp_data/expA1/cnn_7_vs_2.pt' \
    -attack_name='GP_fgsm_eps=20.0_norm=2.0' -action='attack'

