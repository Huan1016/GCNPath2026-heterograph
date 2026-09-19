#!/usr/bin/bash

ic50_th=$1
choice=$2
model=${3:-baseline}

gpu=1
node=0
retrain=0
seed=2021

gnn_cell=4
gnn_drug=2
mode_cell=3
mode_drug=3

case $model in
    baseline) dir_out=RGCN ;;
    heterograph) dir_out=Heterograph ;;
    *) echo "model must be baseline or heterograph"; exit 2 ;;
esac
dir_cell=processed/cell_data_biocarta
dir_drug=processed/drug_data

case $gnn_cell in
    0) cell=${dir_cell}/SANGER_RNA_Lin.pickle ;;
    # *) cell=${dir_cell}/SANGER_RNA_KNN5_Pert2025.pickle ;;
    *) cell=${dir_cell}/SANGER_RNA_KNN5_STR9_Reg_Corr.pickle ;;
esac

case $gnn_drug in
    0) drug=${dir_drug}/GDSC_Drug_Morgan.pickle ;;
    # 0) drug=${dir_drug}/GDSC_Drug_SMILESVec.pickle ;;
    *) drug=${dir_drug}/GDSC_Drug_Graph.pickle ;;
    # *) drug=${dir_drug}/GDSC_Drug_SA.pickle ;;
esac

choice_=(Normal Cell_Blind Drug_Blind Strict_Blind)
dir_out=results/IC50_GDSC_异构图/${choice_[$choice]}/${dir_out}

ic50=data/heterograph_GDSC/response/IC50_GDSC_异构图.txt
data="-cell $cell -drug $drug -ic50 $ic50"
option_attn="-attn_mode 0 -dim_attn 32 -n_attn_layer 1 -coef_ffnn 4 -h_attn 4"
option="-model $model -gnn_cell $gnn_cell -gnn_drug $gnn_drug -mode_cell $mode_cell -mode_drug $mode_drug"
option="$option -n_hid_cell 3 -n_hid_drug 3 -n_hid_pred 2 -act 3"

case $choice in
    3) fold_list=$(seq 0 24) ;;
    *) fold_list=$(seq 0 9) ;;
esac

use_slurm=0
for nth in ${fold_list[@]}
do
    bash train_write.sh "$data" $dir_out $choice $nth \
        "$option" "$option_attn" $gpu $node \
        $ic50_th $seed $retrain $use_slurm
done
