# cuda-mode-hpr

## Environment
```bash
docker build --tag 'cuda-mode-hpr' .
```

## Usage
```bash
docker run --rm -it --entrypoint /bin/bash -v .:/cuda-mode-hpr cuda-mode-hpr
```

## Run
```bash
./cuda_mode_hpr /cuda-mode-hpr/data/example_cloud.pcd /cuda-mode-hpr/data/example_cloud_pose.txt --output /cuda-mode-hpr/data/out.pcd --radius 15000
```