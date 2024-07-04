https://gist.github.com/bll-bobbygill/56afa11759e5f4faa3ac948b10fb3f4b
https://medium.com/@tom_21755/understanding-causal-llms-masked-llm-s-and-seq2seq-a-guide-to-language-model-training-d4457bbd07fa
https://www.bluelabellabs.com/blog/how-to-fine-tune-a-causal-language-model-with-hugging-face/


## Phi3
https://github.com/microsoft/Phi-3CookBook/blob/main/code/04.Finetuning/Phi-3-finetune-qlora-python.ipynb
https://github.com/brevdev/notebooks/blob/main/phi2-finetune-own-data.ipynb

 distillm  --from_pretrained microsoft/Phi-3-mini-4k-instruct           --batch_size 4           --max_steps 2           --eval_steps 2           --no_log           --dataset generic --training_type causal --model_type standard


  distillm  --from_pretrained distilbert/distilgpt2           --batch_size 4           --max_steps 2           --eval_steps 2           --no_log           --dataset generic --training_type causal --model_type standard