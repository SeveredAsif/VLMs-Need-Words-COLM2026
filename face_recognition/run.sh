# run create_dataset.ipynb to create the dataset

python evaluate_gemma_faces.py --dataset_name Qwen2B_FAMOUS_EAST_ASIAN_DATASET --msize 2B --direct True --batch 16 --cuda 0

python evaluate_gemma_faces.py --dataset_name Qwen2B_FAMOUS_EAST_ASIAN_DATASET --msize 2B --direct False --batch 16 --cuda 0

python evaluate_gemma_faces.py --dataset_name FLUXSynID_EAST_ASIAN_DATASET --msize 2B --direct True --batch 16 --cuda 0

python evaluate_gemma_faces.py --dataset_name FLUXSynID_EAST_ASIAN_DATASET --msize 2B --direct False --batch 16 --cuda 0

python rep_probe_gemma.py --dataset_name Qwen2B_FAMOUS_EAST_ASIAN_DATASET --msize 2B --cuda 0

python rep_probe_gemma.py --dataset_name FLUXSynID_EAST_ASIAN_DATASET --msize 2B --cuda 0