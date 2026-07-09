
# 1. Download SPairs-71K from here. https://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
# 2. Optionally upscale the images and annotations in PairAnnotations (recommended to match the paper)
# 3.
# 3. Create dataset with create_sem_corr_dataset.ipynb


python evaluate_sem_corr_qwen.py \
  --model_path Qwen/Qwen3-VL-2B-Instruct \
  --direct true \
  --batch_size 16 \
  --cuda 1 \
  --filters no-name
  
# python evaluate_sem_corr_gemma.py \
#   --model_path google/gemma-3-4b-it \
#   --direct true \
#   --batch_size 4 \
#   --cuda 0 \
#   --filters named