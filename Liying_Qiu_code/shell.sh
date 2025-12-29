#!/bin/sh

###################################################################
################### Section 4: Main results #######################
###################################################################

for ranking_type in "personalized" "unpersonalized2"; do
	python3 main.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
	--step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 --score_type "expected-utility" \
	--n_sum 1000 --ranking_type $ranking_type --start_seed 0 --end_seed 100 --do_summary --do_rational_belief
done

python3 table.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
--step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
--n_sum 1000 --ranking_type "unpersonalized2" --start_seed 0 --end_seed 100 --table_label "baseline" --table_name "baseline" \
--table_caption "Results"

python3 table.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
--step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
--n_sum 1000 --start_seed 0 --end_seed 100 \
--ranking_type "personalized" --score_type "expected-utility" --do_rational_belief --table_name "baseline-rational-belief"

###################################################################
######### Section 5.1: Sales-based unpersonalized ranking #########
###################################################################

python3 main.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
--step_size 0.25 --kappa 1 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 --n_action 10 \
--n_sum 1000 --ranking_type "unpersonalized2" --score_type "sales" --start_seed 0 --end_seed 100 \
--file_prefix "sales" --state "sales" --do_summary

python3 table.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
--step_size 0.25 --kappa 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 --n_action 10 \
--n_sum 1000 --score_type "sales" --start_seed 0 --end_seed 100 --do_rational_belief \
--file_prefix "sales-kappa0.25" --state "sales" --table_label "sales-kappa0.25" --table_name "sales-kappa0.25-rational-belief" \
--table_caption "Results with sales-based unpersonalized ranking"

##################################################################
############# Section 5.2: Learning Schedules ####################
##################################################################

for alpha in 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19; do
  for beta in 1e-6 2e-6 3e-6 4e-6 5e-6 6e-6 7e-6 8e-6 9e-6 1e-5; do
    for ranking_type in "personalized" "unpersonalized2"; do
      python3 main.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
      --step_size 0.25 --alpha $alpha --beta $beta --delta 0.95 --conv_crit 100000 --num_product 2 \
      --n_sum 1000 --ranking_type $ranking_type --start_seed 0 --end_seed 100 --do_summary
    done
  done
done

###################################################################
################# Section 5.3: UCB experiment #####################
###################################################################

for ranking_type in "personalized" "unpersonalized2"; do
	python3 main.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
	--step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 2000000 --n_action 10 --num_product 2 \
	--n_sum 1000 --ranking_type $ranking_type --start_seed 0 --end_seed 100 --learning_algo "UCB_tuned" --do_summary \
    --file_prefix "UCB-conv_crit-2000000" --score_type "expected-utility" --do_rational_belief --do_summary
done

python3 table.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
--step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 2000000 --n_action 10 --num_product 2 \
--n_sum 1000 --start_seed 0 --end_seed 100 --learning_algo "UCB_tuned" --file_prefix "UCB-conv_crit-2000000" \
--score_type "expected-utility" --ranking_type "personalized" --do_rational_belief \
--table_name "UCB-conv_crit-2000000-rational-belief" --table_label "UCB-conv_crit-2000000" \
--table_caption "Outcomes when firms use MAB-UCB algorithms"

###################################################################
################### Section 5.4: Three firms ######################
###################################################################

python3 main.py --a0 0 --a 4 4.25 4.5 --mc 1 1.25 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
--step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 3 --num_customer 10000 --n_sim 1000 \
--n_sum 1000 --ranking_type "personalized3" --start_seed 0 --end_seed 100 --do_summary --do_rational_belief

python3 main.py --a0 0 --a 4 4.25 4.5 --mc 1 1.25 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
--step_size 0.25 --step_size_init 0.2 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 3 --num_customer 10000 --n_sim 1000 \
--n_sum 1000 --ranking_type "unpersonalized3" --score_type "expected-utility" --start_seed 0 --end_seed 100 --do_summary

python3 table.py --a0 0 --a 4 4.25 4.5 --mc 1 1.25 1.5 --mu 1 --ps 1 --sc 1.5 \
--step_size_init 0.2 --conv_crit 100000 --num_product 3 --num_customer 2 --n_sim 4 \
--ranking_type "personalized3" --do_rational_belief --table_name "three_firms_sc1.5-rational-belief" \
--table_caption "Results when there are three firms" --table_label "three_firms"

###################################################################
############## Section 5.5: Imperfect personalization #############
###################################################################

for lamb in 0.5 0.6 0.7 0.8 0.9 1.0; do
    python3 main.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb $lamb --xi 0.1 \
    --step_size 0.15 --step_size_init 0.1 --num_customer 5000 \
    --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
    --n_sum 1000 --ranking_type "imperfect-rational-n_customer5000" --start_seed 0 --end_seed 100 --do_summary \
    --file_prefix "lambda$lamb" 
done

###################################################################
################## Section 5.6: Search cost #######################
###################################################################

for sc in 1 2; do
    python3 main.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc $sc --gamma -1 --lamb 1 --xi 0.1 \
    --step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
    --n_sum 1000 --ranking_type "personalized" --start_seed 0 --end_seed 100 --do_summary \
    --do_summary
done

for sc in 1 2; do
    python3 main.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc $sc --gamma -1 --lamb 1 --xi 0.1 \
    --step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
    --n_sum 1000 --ranking_type "unpersonalized" --start_seed 0 --end_seed 100 --do_summary \
    --do_summary
done

for sc in 1 1.5 2; do
    python3 table.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc $sc --gamma -1 --lamb 1 --xi 0.1 \
    --step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
    --n_sum 1000 --start_seed 0 --end_seed 100 --table_label "sc$sc" --table_name "sc$sc" \
    --table_caption "Results when search cost is $sc"
done

for sc in 1 2; do
    python3 table.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc $sc --gamma -1 --lamb 1 --xi 0.1 \
    --step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
    --n_sum 1000 --start_seed 0 --end_seed 100 --table_label "sc$sc" --table_name "sc$sc-rational-belief" \
    --table_caption "Results when search cost is $sc" --do_rational_belief --ranking_type "personalized"
done

###################################################################
########## Section 5.7: Different outside goods value #############
###################################################################

for a0 in -1 1; do
    python3 main.py --a0 $a0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
    --step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
    --n_sum 1000 --ranking_type "unpersonalized2" --score_type "expected-utility" --start_seed 0 --end_seed 100 --do_summary \
    --do_summary
done

for a0 in -1 1; do
    python3 main.py --a0 $a0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
    --step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
    --n_sum 1000 --ranking_type "personalized" --start_seed 0 --end_seed 100 --do_summary \
    --do_summary
done

for a0 in -1 1; do
    python3 table.py --a0 $a0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
    --step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
    --n_sum 1000 --ranking_type "personalized" --start_seed 0 --end_seed 100 \
    --do_rational_belief --table_label "a0$a0" --table_name "a0$a0-rational-belief" --table_caption "Results when vertical differentiation"
done

for a0 in -1 1; do
    python3 table.py --a0 $a0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
    --step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
    --n_sum 1000 --start_seed 0 --end_seed 100 --table_label "a0$a0" --table_name "a0$a0" \
    --table_caption "Results when vertical differentiation"
done

###################################################################
############## Section A.7: Risk averse consumers #################
###################################################################

for ranking_type in "personalized" "unpersonalized2"; do 
    python3 main.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
    --step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
    --r 0.025 --n_sim 1000 --num_customer 1000 --step_size_init 0.1 --score_type "risk-averse" --file_prefix "risk-averse-0.025" \
    --n_sum 1000 --ranking_type $ranking_type --start_seed 0 --end_seed 100 --do_summary --do_rational_belief
done

python3 table.py --a0 0 --a 4 4.5 --mc 1 1.5 --mu 1 --ps 1 --sc 1.5 --gamma -1 --lamb 1 --xi 0.1 \
--step_size 0.25 --alpha 0.1 --beta 2e-6 --delta 0.95 --conv_crit 100000 --num_product 2 \
--r 0.025 --n_sim 1000 --num_customer 1000 --step_size_init 0.1 --score_type "risk-averse" --file_prefix "risk-averse-0.025" \
--n_sum 1000 --start_seed 0 --end_seed 100 --table_label "risk-averse" --table_name "risk-averse-rational-belief" \
--table_caption "Results when consumers are risk averse"
