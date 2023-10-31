if [ -z $DATA ]; then
    DATA=custom
fi

if [ -z $VIDEO ]; then
    VIDEO=custom.mp4
fi

if [ -z $H ]; then
    H=0
fi

if [ -z $W ]; then
    W=0
fi

if [ -z $NUM_DATA ]; then
    NUM_DATA=100
fi

if [ ! -f "data/$DATA/original.mp4" ]; then
    if [ ! -f $VIDEO ]; then
        echo "$VIDEO doesn't exists"
        exit 1
    fi
fi

if [ ! -d "data/$DATA" ]; then
    mkdir -p data/$DATA
fi

###### 1. Copy Original Video to the data directory ######
if [ ! -f "data/$DATA/original.mp4" ]; then
    cp $VIDEO data/$DATA/original.mp4
fi
##########################################################


########### 2. Unpack all frames of the video ############
if [ ! -d "data/$DATA/original_images" ]; then
    mkdir -p data/$DATA/temp_original_images
    ffmpeg -i data/$DATA/original.mp4 data/$DATA/temp_original_images/%04d.png
    mv data/$DATA/temp_original_images data/$DATA/original_images
fi
##########################################################


##### 3. Uniformly Sample, Crop and Resize the Image #####
if [ ! -d "data/$DATA/images" ]; then
    mkdir -p data/$DATA/temp_images
    python scripts/crop_and_resize.py \
    --src_root data/$DATA/original_images \
    --dst_root data/$DATA/temp_images \
    --H $H \
    --W $W \
    --num_data $NUM_DATA
    mv data/$DATA/temp_images data/$DATA/images
fi
##########################################################


######## 4. Run Colmap and Convert to Text Format ########
if [ ! -f "data/$DATA/colmap_text/images.txt" ]; then
    colmap automatic_reconstructor \
    --workspace_path data/$DATA \
    --image_path data/$DATA/images \
    --camera_model OPENCV \
    --single_camera 1 \
    --sparse 1 \
    --dense 0

    mkdir data/$DATA/colmap_text
    
    colmap model_converter \
    --input_path data/$DATA/sparse/0 \
    --output_path data/$DATA/colmap_text \
    --output_type TXT
fi
##########################################################


############# 5. Preprocess ###############
if [ ! -f "data/$DATA/transforms.json" ]; then
    cd data/$DATA
    python ../../scripts/colmap2nerf.py \
    --images images \
    --text colmap_text \
    --out transforms.json \
    --overwrite
    cd ../..
fi
##########################################################