if [ -z $DATA ]; then
    DATA=custom
fi

if [ -z $VIDEO ]; then
    VIDEO=custom.mp4
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
    mv $VIDEO data/$DATA/original.mp4
fi
##########################################################


########### 2. Unpack all frames of the video ############
if [ ! -d "data/$DATA/original_images" ]; then
    mkdir -p data/$DATA/temp
    ffmpeg -i data/$DATA/original.mp4 data/$DATA/temp/%04d.png
    mv data/$DATA/temp data/$DATA/original_images
fi
##########################################################


##### 3. Uniformly Sample, Crop and Resize the Image #####
if [ ! -d "data/$DATA/images" ]; then
    mkdir -p data/$DATA/temp
    python scripts/crop_and_resize.py \
    --src_root data/$DATA/original_images \
    --dst_root data/$DATA/temp \
    --num_data 200
    mv data/$DATA/temp data/$DATA/images
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
    python scripts/colmap2nerf.py \
    --images data/$DATA/images \
    --text data/$DATA/colmap_text \
    --out data/$DATA/transforms.json \
    --overwrite
fi
##########################################################