python3 cnn.py -attack='fgsm' -eps=0.01 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.05 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.06 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.065 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.07 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.075 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.08 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.1 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.15 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.2 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.25 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.3 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.35 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.4 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.45 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=0.5 -report='cnn_fgsm_inf_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=1 -ord=2 -report='cnn_fgsm_2_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=2 -ord=2 -report='cnn_fgsm_2_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=5 -ord=2 -report='cnn_fgsm_2_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=10 -ord=2 -report='cnn_fgsm_2_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=20 -ord=2 -report='cnn_fgsm_2_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=1 -ord=1 -report='cnn_fgsm_1_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=2 -ord=1 -report='cnn_fgsm_1_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=5 -ord=1 -report='cnn_fgsm_1_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=10 -ord=1 -report='cnn_fgsm_1_arch0.txt'
python3 cnn.py -attack='fgsm' -eps=20 -ord=1 -report='cnn_fgsm_1_arch0.txt'
python3 cnn.py -attack='ead' -confidence=2.0 -report='cnn_ead_arch0.txt' -max_iterations=10 -learning_rate=0.5 -initial_const=10.0 -beta=0.25
python3 cnn.py -attack='ead' -confidence=0.0 -report='cnn_ead_arch0.txt' -max_iterations=10 -learning_rate=0.5 -initial_const=10.0 -beta=0.25
python3 cnn.py -attack='cw_l2' -confidence=2.0 -report='cnn_cw_l2_arch0.txt' -max_iterations=10 -learning_rate=0.5 -initial_const=10.0
python3 cnn.py -attack='cw_l2' -confidence=0.0 -report='cnn_cw_l2_arch0.txt' -max_iterations=10 -learning_rate=0.5 -initial_const=10.0
python3 cnn.py -attack='pgd' -report='cnn_pgd_arch0.txt' -nb_iter=10 -eps=0.01 eps_iter=0.001 
python3 cnn.py -attack='pgd' -report='cnn_pgd_arch0.txt' -nb_iter=10 -eps=0.05 eps_iter=0.005
python3 cnn.py -attack='pgd' -report='cnn_pgd_arch0.txt' -nb_iter=10 -eps=0.1 eps_iter=0.01
python3 cnn.py -attack='pgd' -report='cnn_pgd_arch0.txt' -nb_iter=10 -eps=0.2 eps_iter=0.02
python3 cnn.py -attack='pgd' -report='cnn_pgd_arch0.txt' -nb_iter=10 -eps=0.3 eps_iter=0.03
python3 cnn.py -attack='pgd' -report='cnn_pgd_arch0.txt' -nb_iter=10 -eps=0.4 eps_iter=0.04
python3 cnn.py -attack='pgd' -report='cnn_pgd_arch0.txt' -nb_iter=10 -eps=0.5 eps_iter=0.05
python3 cnn.py -attack='pgd' -report='cnn_pgd_arch0.txt' -nb_iter=10 -eps=1 eps_iter=0.1
python3 cnn.py -attack='pgd' -report='cnn_pgd_arch0.txt' -nb_iter=10 -eps=2 eps_iter=0.2
python3 cnn.py -attack='pgd' -report='cnn_pgd_arch0.txt' -nb_iter=10 -eps=5 eps_iter=0.5
python3 cnn.py -attack='pgd' -report='cnn_pgd_arch0.txt' -nb_iter=10 -eps=10 eps_iter=1
python3 cnn.py -attack='pgd' -report='cnn_pgd_arch0.txt' -nb_iter=10 -eps=20 eps_iter=2.0
