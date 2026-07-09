python generate_dataset.py --input_folder squiggles_30 --output_folder squiggles_30_TEST --num_tasks 100

python generate_dataset.py --input_folder basic_shapes --output_folder basic_shapes_TEST --num_tasks 100

python evaluate_qwen_squiggles.py --dataset squiggles_30_TEST --cuda 0 --batch 8 --direct True --model_path Qwen/Qwen3-VL-2B-Instruct

python rep_probe_qwen.py --dataset squiggles_30_TEST --cuda 0 --model_path Qwen/Qwen3-VL-2B-Instruct
