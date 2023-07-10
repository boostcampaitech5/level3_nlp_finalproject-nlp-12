# 학습한 lora를 원래 모델에 합치는 기능
# base_model_path : 원래 백본 모델
# lora_path : qlora로 학습이 완료된 adapter_model.bin, adapter_config.json 경로
# target_model_path : 합친 모델 저장할 경로

python3 ./utils/merge_lora.py \
    --base_model_path 'nlpai-lab/kullm-polyglot-12.8b-v2' \
    --lora_path ./model/lora_finetuned/test \
    --target_model_path ./model/lora_merged/test \