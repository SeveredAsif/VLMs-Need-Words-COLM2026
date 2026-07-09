python create_painting_dataset.py --model_name Qwen/Qwen3-VL-8B-Instruct --num_samples 500

python evaluate_faces.py --dataset_name known_paintings_dataset/ --msize 8B --direct True --batch 16 --cuda 0
python evaluate_faces.py --dataset_name unknown_paintings_dataset/ --msize 8B --direct True --batch 16 --cuda 0
python evaluate_faces.py --dataset_name known_paintings_dataset/ --msize 8B --direct False --batch 16 --cuda 0
python evaluate_faces.py --dataset_name unknown_paintings_dataset/ --msize 8B --direct False --batch 16 --cuda 0

python rep_probe_qwen.py --dataset_name known_paintings_dataset --msize 8B --cuda 0
python rep_probe_qwen.py --dataset_name unknown_paintings_dataset --msize 8B --cuda 0
