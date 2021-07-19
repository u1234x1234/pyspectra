FROM rapidsai/rapidsai-core:21.06-cuda11.2-runtime-ubuntu18.04-py3.8

COPY benchmark_svd_cuml.py ./
CMD [ "bash" ]
