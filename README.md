Install tensorflow optimized
    
    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    ./configure
    bazel build -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 -k //tensorflow/tools/pip_package:build_pip_package