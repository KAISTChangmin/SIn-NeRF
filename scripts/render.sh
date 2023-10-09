if [ -z $DATA ]; then
    DATA=farm-small
fi
if [ -z $EXPNAME ]; then
    EXPNAME=farm_desert
fi

RENDER_NERF=true
RENDER_IN2N=true


if [ -d "outputs/$DATA/nerfacto" ]; then
    NERF_TIMESTAMP=$(ls outputs/$DATA/nerfacto | tail -n1)
    if [ ! -z $NERF_TIMESTAMP ]; then
        NERF_DIR=outputs/$DATA/nerfacto/$NERF_TIMESTAMP
    fi
fi


#################### NeRF Rendering ###################
if $RENDER_NERF; then
    ns-render interpolate \
    --load-config $NERF_DIR/config.yml \
    --output-path renders/$DATA \
    --pose-source train \
    --interpolation-steps 1 \
    --output-format images
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
    ns-render interpolate \
    --load-config $IN2N_DIR/config.yml \
    --output-path renders/$EXPNAME \
    --pose-source train \
    --interpolation-steps 1 \
    --output-format images
fi
#######################################################