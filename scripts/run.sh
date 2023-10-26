if [ -z $DATA ]; then
    DATA=js
fi
if [ -z $EXPNAME ]; then
    EXPNAME=js_clown
fi
if [ -z "$PROMPT" ]; then
    PROMPT="Turn him into a clown"
fi
if [ -z $GUIDANCE_SCALE ]; then
    GUIDANCE_SCALE=7.5
fi
if [ -z $IMAGE_GUIDANCE_SCALE ]; then
    IMAGE_GUIDANCE_SCALE=1.5
fi
if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES="0,1,2,3"
fi
if [ -z $NUM_DEVICES ]; then
    NUM_DEVICES=1
fi
if [ -z $PORT ]; then
    PORT=7007
fi


TRAIN_NERF=false
RENDER_NERF=false
TRAIN_IN2N=false
RENDER_IN2N=false
CONCAT_VIDEO=false
STACK=hstack            # vstack (vertical) / hstack (horizontal)

#################### NeRF Training ####################
if $TRAIN_NERF; then
    ns-train nerfacto \
    --data data/$DATA \
    --max-num-iterations 30000 \
    --machine.num-devices $NUM_DEVICES \
    --viewer.websocket-port $PORT \
    --viewer.quit-on-train-completion True \
    nerfstudio-data --train-split-fraction 1
fi
#######################################################


if [ -d "outputs/$DATA/nerfacto" ]; then
    NERF_TIMESTAMP=$(ls outputs/$DATA/nerfacto | tail -n1)
    if [ ! -z $NERF_TIMESTAMP ]; then
        NERF_DIR=outputs/$DATA/nerfacto/$NERF_TIMESTAMP
    fi
fi


#################### NeRF Rendering ###################
if $RENDER_NERF; then
    ns-render camera-path \
    --load-config $NERF_DIR/config.yml \
    --output-path renders/$DATA.mp4 \
    --camera-path-filename data/$DATA/camera_paths/final-path.json
fi
#######################################################


#################### IN2N Training ####################
if $TRAIN_IN2N; then
    ns-train in2n \
    --data data/$DATA \
    --load-dir $NERF_DIR/nerfstudio_models \
    --experiment-name $EXPNAME \
    --pipeline.prompt "$PROMPT" \
    --pipeline.guidance-scale $GUIDANCE_SCALE \
    --pipeline.image-guidance-scale $IMAGE_GUIDANCE_SCALE \
    --max-num-iterations 15000 \
    --machine.num-devices $NUM_DEVICES \
    --viewer.websocket-port $PORT \
    --viewer.quit-on-train-completion True \
    nerfstudio-data --train-split-fraction 1
fi
#######################################################


if [ -d "outputs/$EXPNAME/in2n" ]; then
    IN2N_TIMESTAMP=$(ls outputs/$EXPNAME/in2n | tail -n1)
    if [ ! -z $IN2N_TIMESTAMP ]; then
        IN2N_DIR=outputs/$EXPNAME/in2n/$IN2N_TIMESTAMP
    fi
fi


#################### IN2N Rendering ###################
if $RENDER_IN2N; then
    ns-render camera-path \
    --load-config $IN2N_DIR/config.yml \
    --output-path renders/$EXPNAME.mp4 \
    --camera-path-filename data/$DATA/camera_paths/final-path.json
fi
#######################################################


#################### Concat Videos ####################
if $CONCAT_VIDEO; then
    ffmpeg \
    -i renders/$DATA.mp4 \
    -i renders/$EXPNAME.mp4 \
    -filter_complex \
    $STACK \
    -y \
    renders/${EXPNAME}_concat.mp4
fi
#######################################################