set -e

#if [ $# -ne 1 ]; then
#  echo "usage: $0 <tensorflow-root-dir>" >&2
#  exit 1
#fi
#
#rsync --existing "$1"/tensorflow/core/framework/*.proto tensorboard/compat/proto/
#rsync --existing "$1"/tensorflow/core/protobuf/*.proto tensorboard/compat/proto/
#rsync --existing "$1"/tensorflow/core/profiler/*.proto tensorboard/compat/proto/
#rsync --existing "$1"/tensorflow/core/util/*.proto tensorboard/compat/proto/
#rsync --existing "$1"/tensorflow/python/framework/*.proto tensorboard/compat/proto/
#
## Rewrite file paths and package names.
#find tensorboard/compat/proto/ -type f  -name '*.proto' -exec perl -pi \
#  -e 's|tensorflow/core/framework|tensorboard/compat/proto|g;' \
#  -e 's|tensorflow/core/protobuf|tensorboard/compat/proto|g;' \
#  -e 's|tensorflow/core/profiler|tensorboard/compat/proto|g;' \
#  -e 's|tensorflow/core/util|tensorboard/compat/proto|g;' \
#  -e 's|tensorflow/python/framework|tensorboard/compat/proto|g;' \
#  -e 's|package tensorflow.tfprof;|package tensorboard;|g;' \
#  -e 's|package tensorflow;|package tensorboard;|g;' \
#  -e 's|tensorflow\.DataType|tensorboard.DataType|g;' \
#  -e 's|tensorflow\.TensorShapeProto|tensorboard.TensorShapeProto|g;' \
#  {} +
#
#echo "Protos in tensorboard/compat/proto/ updated! You can now add and commit them."

find src/tensorboardrs/proto/ -type f  -name '*.proto' -exec perl -pi \
     -e 's|package tensorboardX;|package tensorboardrs;|g;' \
     -e 's|tensorboardX/proto/|src/tensorboardrs/proto/|g;' \
     -e 's|package tensorboardX.|package tensorboardrs.|g;' \
  {} +
