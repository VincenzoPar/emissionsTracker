import accelerate
import transformers
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM

tracker = EmissionsTracker(project_name="LLama2I_Inference")
model_id = "meta-llama/Llama-2-7b-chat-hf"
config = transformers.AutoConfig.from_pretrained(model_id)

with accelerate.init_empty_weights():
    fake_model = transformers.AutoModelForCausalLM.from_config(config)

device_map = accelerate.infer_auto_device_map(fake_model, max_memory={0: "3GiB", "cpu": "10GiB"})
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
      model_id,
      device_map=device_map,
      load_in_8bit=True,
      llm_int8_enable_fp32_cpu_offload=True
)

text = "Laughter is the best medicine for the soul"
device = "cuda:0"

inputs = tokenizer(text, return_tensors="pt").to(device)
tracker.start()
try:
    outputs = model.generate(**inputs, max_new_tokens=300)
finally:
    tracker.stop()

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
