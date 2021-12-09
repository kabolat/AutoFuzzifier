#!/bin/sh


datasets=("abalone" "airfoil" "autompg" "concrete" "no2" "pm10" "powerplant" "protein" "redwine" "servo" "whitewine")
models=("fuzzy" "vanilla")
fcm_loss=1
SVD=1
latent_dim=5
num_clusters=10
epochs=(300 300 400)
loss_coeffs=(10 0.1 1)
batch_size=128
learning_rate=1e-4


for dataset in ${datasets[*]};
do
	for model in ${models[*]};
	do
		echo $dataset
		echo $model
		python main.py --dataset $dataset --model $model --fcm_loss $fcm_loss --SVD $SVD --latent_dim $latent_dim \
		 		--num_clusters $num_clusters --epochs ${epochs[*]} --loss_coeffs $loss_coeffs --batch_size $batch_size --learning_rate $learning_rate
	done
done
