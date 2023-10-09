if [ -z $DATA ]; then
    DATA=custom
fi

cd data/$DATA

python ../../scripts/colmap2nerf.py \
--run_colmap \
--images ./images/ \
--out ./transforms.json \
--overwrite

cd ../..