declare -a dataset=(mvtec visa)
save_path="./TESTING_ALL"
for i in "${dataset[@]}"; do
    python test.py --result_path $save_path --dataset $i
done