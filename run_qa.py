import json
import time
import os
import torch
import pandas as pd
from transformers import AutoTokenizer, LlamaForCausalLM, pipeline
from transformers import BitsAndBytesConfig

def find_diagnosis_info(decision, comment):
    # Filter the dataframe for rows matching the provided diagnosis (decision)
    match_row = dx_data[(dx_data['Dx (Korean)'].str.lower() == decision.lower()) 
    & (dx_data['Original Comment (Korean)'].str.lower() == comment.lower())]
    
    if not match_row.empty:
        # Extract the required fields from the matched row
        explanation = match_row.iloc[0]['Explanation']
        causes = match_row.iloc[0]['Causes']
        symptoms = match_row.iloc[0]['Sx']
        treatment = match_row.iloc[0]['Tx']
        
        # Print the extracted information
        print(f"Diagnosis Explanation: {explanation}")
        print()
        print(f"Causes: {causes}")
        print()
        print(f"Symptoms: {symptoms}")
        print()
        print(f"Treatment: {treatment}")
    else:
        print("No matching diagnosis found in the data.")

def return_diagnosis(decision, comment):
    # Filter the dataframe for rows matching the provided diagnosis (decision)
    match_row = dx_data[(dx_data['Dx (Korean)'].str.lower() == decision.lower()) 
    & (dx_data['Original Comment (Korean)'].str.lower() == comment.lower())]
    
    if not match_row.empty:
        # Extract the required fields from the matched row
        explanation = match_row.iloc[0]['Dx (English)']
        cause = match_row.iloc[0]['Causes']
        symptom = match_row.iloc[0]['Sx']
        treatment = match_row.iloc[0]['Tx']
        
        return explanation, cause, symptom, treatment

    else:
        print("No matching diagnosis found in the data.")

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

root_dir = './data'
dx_data = pd.read_csv(os.path.join(root_dir, 'dx_data.csv'))

# Start timer
start_time = time.time()

# Clear CUDA cache
torch.cuda.empty_cache()

# Use the 'expandable_segments' option to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

################################################### Pipieline ##############################################

# Load the saved model and tokenizer
model_id = "./local_model/med42_bitsandbytes_4bit"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Configure quantization for 4-bit with bitsandbytes
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float16
)

# Load the model with optimizations
model = LlamaForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    quantization_config=quantization_config,
    device_map="auto"
)

# Create a pipeline for text generation
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.5,
    device_map="auto"
)

############################################## Prompt Engineering ############################################

# Define the prompt template
template = """You are a Medical Language Model tasked with providing a clear
and patient-friendly explanation of a medical diagnosis obtained through an Esophagogastroduodenoscopy (EGD), as well as personalized lifestyle recommendations. Consider the patient's age, sex, and specific diagnosis in your response.

Objective:
1. Customized health advice will be provided in consideration of the patient's inputs: age, sex, diagnosis, comment, explanation, causes, symptoms, and treatment plan. There should be four factors in this: risk of disease, reassuring and empathizing the diagnosed patient, eating habits based on the disease the patient is diagnosed with, and good exercise based on the disease the patient is diagnosed with.
2. All four of the above factors should be described in detailed and friendly terms that the patient can easily understand based on the condition diagnosed in EGD.
3. Don't add anything other than the above four elements

Input:
- Age: {age}
- Sex: {sex}
- Diagnosis: {diagnosis}
- Cause: {cause}
- Symptom: {symptom}
- Treatment: {treatment}

Output:
1. Risk of disease: The risk of disease should present the risk for the disease diagnosed by the patient. In particular, the gender and age of the patient should be considered. For example, the risk of disease in general will increase due to decreased gastrointestinal function and accompanying chronic diseases as they are older. In addition, more specifically, for example, in the case of acute gastritis, the risk of acute gastritis will be high in young people due to frequent drinking, irregular eating habits, and stress. In addition, men will have a high risk of acute gastritis due to factors such as smoking, drinking, and stress, and women may have a high risk of acute gastritis due to factors such as hormonal changes, pregnancy, and childbirth. In this way, the risk of considering age or gender should be presented for each disease.
2. Reassuring & compassionation: Give compassionate and reassuring comment for patient diagnosed condition with "Diagnosis".
e.g.) I told you that gastritis can develop into stomach cancer, but the possibility is significantly less, so if you pay attention to your lifestyle from now on, it can be prevented enough. ^^
3. Eating habits: It presents detailed eating habits that should be good for the patient to have according to the disease diagnosed by the patient, or that the patient should abandon.
4. Exercise: The exercise habits that would be good for the patient to have according to the disease diagnosed by the patient, and the exercise that is helpful are solved along with the exercise method.


Now, create a response following this structure based on the given inputs.
"""

# Define the function to generate an answer
def generate_answer(age, sex, diagnosis, cause, symptom, treatment):
    prompt = template.format(age=age, sex=sex, diagnosis=diagnosis, cause=cause, symptom=symptom, treatment=treatment)
    response = pipe(prompt)
    
    # Extract the answer part from the generated text
    generated_text = response[0]['generated_text']
    
    # Remove the input prompt from the generated output
    answer = generated_text[len(prompt):].strip() 
    return answer

# Single Example input for inference
age = 45
sex = "Male"
decision = "크기가 큰 위점막하종양"
comment = "위내시경검사에서 위에 점막하종양이 있습니다. 점막하종양이란 위점막 아래에서 발생한 종양을 말하며, 이 때문에 위벽 일부가 부풀어 올라온 것처럼 보입니다. 점막하 종양은 종류에 따라 아무런 치료가 필요하지 않는 경우부터 수술적 절제가 필요한 경우까지 치료방법이 매우 다양합니다. 정확한 진단 및 적절한 치료 방침 결정을 위해 의료기관을 방문하시어 진료 상담을 받으시길 바랍니다."
diagnosis, cause, symptom, treatment = return_diagnosis(decision, comment)

# Generate answer for the single input
answer = generate_answer(age, sex, diagnosis, cause, symptom, treatment)


######################################### Report Generation Section ##################################

# Print the result
print(f"Age: {age}")
print(f"Sex: {sex}")

# Find english diagnosis
find_diagnosis_info(decision, comment)

# Run the function with example input
print()
print(f"Generated Text: {answer}")

# End timer
end_time = time.time()
print('Total elapsed time: ', end_time - start_time)
