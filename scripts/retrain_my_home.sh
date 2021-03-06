#!/bin/bash

BOTTLENECK_FOLDER="../tf_files/bottleneck"
GRAPH_DEF_FILE="../tf_files/retrain_model.pb"
OUT_TFLITE_FILE="../tf_files/mobilenet_v1_224_retrain.tflite"
OUT_LABLE_RETRAIN="../tf_files/labels_mobilenet_v1_224_retrain.txt"
RETRAIN_MODEL="mobilenet_1.0_224"
RETRAIN_DATA="./my_home"
IMAGE_SIZE=224

#Clear bottleneck folder
echo "clear bottleneck folder ${BOTTLENECK_FOLDER}"
rm -rf ${BOTTLENECK_FOLDER}
rm -rf ${GRAPH_DEF_FILE}
rm -rf ${OUT_LABLE_RETRAIN}
rm -rf ${OUT_TFLITE_FILE}

#Copy original model to /tmp/imagenet
ORIGINAL_TF_MODEL="/tmp/imagenet/mobilenet_v1_1.0_224_frozen.tgz"
if [ -f "$ORIGINAL_TF_MODEL" ]; then
    echo "Original TensorFlow Model Found ==> /tmp/imagenet/mobilenet_v1_1.0_224_frozen.tgz exist"
else
    mkdir -p /tmp/imagenet
    cp ./mobilenet_v1_1.0_224_frozen.tgz /tmp/imagenet
    tar xvzf /tmp/imagenet/mobilenet_v1_1.0_224_frozen.tgz /tmp/imagenet/
fi

#Retrain model
python retrain.py --image_dir ${RETRAIN_DATA} --output_graph=${GRAPH_DEF_FILE} --output_labels=${OUT_LABLE_RETRAIN} --architecture=${RETRAIN_MODEL} --bottleneck_dir=${BOTTLENECK_FOLDER}

# Convert *.pb to *.tflite
toco \
--graph_def_file=${GRAPH_DEF_FILE} \
--output_file=${OUT_TFLITE_FILE} \
--input_format=TENSORFLOW_GRAPHDEF \
--output_format=TFLITE \
--input_shape=1,${IMAGE_SIZE},${IMAGE_SIZE},3 \
--input_array=input \
--output_array=final_result \
--inference_type=FLOAT \
--input_data_type=FLOAT

echo "Output retrain label file: ${OUT_LABLE_RETRAIN}"
echo "Output tflite file:  ${OUT_TFLITE_FILE}"
