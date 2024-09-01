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

def get_optional_inputs(history=None, family_history=None, past_procedure=None, past_infection=None, obesity=None,
                        coffee=None, alcohol=None, smoking=None, overeating=None, spicy_salting_fatty_food=None, exercise=None):
    
    optional_inputs = {
        "history": bool(history) if history is not None else None,
        "family_history": bool(family_history) if family_history is not None else None,
        "past_procedure": bool(past_procedure) if past_procedure is not None else None,
        "past_infection": bool(past_infection) if past_infection is not None else None,
        "obesity": bool(obesity) if obesity is not None else None,
        "coffee": bool(coffee) if coffee is not None else None,
        "alcohol": bool(alcohol) if alcohol is not None else None,
        "smoking": bool(smoking) if smoking is not None else None,
        "overeating": bool(overeating) if overeating is not None else None,
        "spicy_salting_fatty_food": bool(spicy_salting_fatty_food) if spicy_salting_fatty_food is not None else None,
        "exercise": bool(exercise) if exercise is not None else None
    }

    return optional_inputs

################################################ Manual template making ########################################

def make_template(age, sex, diagnosis, cause, symptom, treatment, optional_inputs):
    
    # Base template
    base_template = """
    You are a highly specialized customized health adviser. Your primary role is to provide personalized health advice for each patient based on the information provided. Your responses must take into account the patient's specific details to ensure that the advice is relevant, compassionate, and easy to understand.

    ### Instructions:

    1. Persona: Act as a customized health adviser. Your responses must be empathetic and patient-focused, delivering health information in a friendly and supportive tone.

    2. Role: Your main goal is to provide personalized health advice tailored to the patient's needs. Base your advice on the patient's general and specific information, considering their medical history, habits, and any provided diagnosis.

    3. Input: The input will consist of three sections:
    - General Information: Includes the patient's age and sex.
    - Specific Information (Optional): May include details such as family history, past medical interventions, lifestyle habits (e.g., smoking, diet, obesity), and other relevant factors.
    - Diagnosis and Comment: Provides the patient's diagnosis, causes, symptoms, and potential treatment options.

    4. Output: Your response will be structured into four key sections:
    - Risk of Disease: Provide an assessment of the patient's disease risk, including personalized information where possible.
    - Reassurance and Compassion: Offer a comforting, empathetic message that reassures the patient, using simple, easily understood language.
    - Eating Habits for Diagnosis: Advise on dietary habits that may help the patient’s condition, considering their diagnosis and medical history.
    - Exercise for Diagnosis: Recommend appropriate physical activity that aligns with the patient's diagnosis and overall health.

    5. Overall Prompt Composition: After receiving (1) the patient's input, we will provide (2) how to interpret the information carefully, factoring in both general and specific details. Also, we will provide (3) the detailed instructions for each section of the output and (4) how to structure your response.

    6. Warning: Personalization is critical. Each response must reflect the patient's unique circumstances. Avoid using complex medical jargon. Focus on delivering information in a friendly, accessible, and compassionate manner that the patient can easily understand.
    
    You should consider the following sections for each input response and generate a personalized health advice based on the patient's information:

    General Information

    - age: {age}: "Older patients are generally at a higher risk of developing gastritis and stomach cancer. For instance, older patients may generally have a higher risk due to factors like decreased digestive function. Therefore, you should write the response considering the patient's age."
    - sex: {sex}: "Males are generally at a higher risk of developing stomach cancer than female. Therefore, you should write the response considering the patient's sex."

    - Diagnosis: {diagnosis}
    - Cause: {cause}
    - Symptom: {symptom}
    - Treatment: {treatment}
    """

    # Start building the optional inputs section
    optional_template = "\nOptional input: these are optional inputs of the patient. Please consider this optional information if needed. You should print these information in main output template, so don't make other sections for optional information. If 'None', you must not consider it.\n"
    
    # Check and append optional inputs if they are not None, including their boolean value and context description
    if optional_inputs["history"] is not None:
        optional_template += f"    - history (bool: {optional_inputs['history']}): Previous diagnosis status. If the previous diagnosis included gastritis, atrophic gastritis, or intestinal metaplasia, and these conditions have either persisted or progressed, there is a very high risk of developing stomach cancer, so it is important to be cautious.\n"
    if optional_inputs["family_history"] is not None:
        optional_template += f"    - family_history (bool: {optional_inputs['family_history']}): Family history of stomach cancer. If there is a family history of stomach cancer, you need to be even more cautious about the risk of developing it.\n"
    if optional_inputs["past_procedure"] is not None:
        optional_template += f"    - past_procedure (bool: {optional_inputs['past_procedure']}): History of upper gastrointestinal procedures/surgeries. If you have undergone any upper gastrointestinal procedures or surgeries, you need to be more cautious about the risk of gastritis and stomach cancer.\n"
    if optional_inputs["past_infection"] is not None:
        optional_template += f"    - past_infection (bool: {optional_inputs['past_infection']}): History of H. pylori infection. If you have a history of Helicobacter pylori infection, you need to be more cautious about the risk of gastritis and stomach cancer.\n"
    if optional_inputs["obesity"] is not None:
        optional_template += f"    - obesity (bool: {optional_inputs['obesity']}): Whether the patient is obese. If obesity persists, it becomes a risk factor for stomach cancer.\n"
    if optional_inputs["coffee"] is not None:
        optional_template += f"    - coffee (bool: {optional_inputs['coffee']}): Consumption of coffee. Consumption of coffee can be causes of gastritis.\n"
    if optional_inputs["alcohol"] is not None:
        optional_template += f"    - alcohol (bool: {optional_inputs['alcohol']}): Consumption of alcohol. Excessive alcohol consumption can cause gastritis.\n"
    if optional_inputs["smoking"] is not None:
        optional_template += f"    - smoking (bool: {optional_inputs['smoking']}): Smoking can cause both gastritis and stomach cancer. Notably, smoking is very important factor in the development of stomach cancer.\n"
    if optional_inputs["overeating"] is not None:
        optional_template += f"    - overeating (bool: {optional_inputs['overeating']}): Whether the patient practices binge eating or overeating. Overeating or binge eating can cause both gastritis and stomach cancer.\n"
    if optional_inputs["spicy_salting_fatty_food"] is not None:
        optional_template += f"    - spicy_salting_fatty_food (bool: {optional_inputs['spicy_salting_fatty_food']}): Consumption of spicy, salty, or fatty foods. Consuming spicy, salty, or fatty foods can be contributing factors to both gastritis and stomach cancer.\n"
    if optional_inputs["exercise"] is not None:
        optional_template += f"    - exercise (bool: {optional_inputs['exercise']}): Whether the patient exercises or not. If you are not exercising regularly, aim for at least 30 minutes of exercise, 5 times a week, to help prevent cancer. However, if you are already exercising, be cautious not to engage in vigorous activities that might strain the abdominal area.\n"
    
    # Add output description
    output_template = """
    ### Output Description:

    1. Risk of Disease: Assess and explain the patient's risk for the condition, taking into account their diagnosis and personal information. Offer personalized insights where possible.
    2. Reassuring & Compassion: Provide comforting and empathetic words that address the patient’s concerns about their diagnosed condition. Offer reassurance to help ease their worries.
    3. Eating Habits for Diagnosis: Recommend dietary habits that are beneficial for managing the diagnosed condition. Suggest specific foods to include or avoid, and provide detailed advice on how to modify eating habits for better health outcomes.
    4. Exercise for Diagnosis: Suggest exercise habits that align with the patient’s diagnosis. Explain the types of physical activities that would be most beneficial, including how to perform these exercises safely and effectively.


    ### Output Format:

    You should provide a response that includes the following sections:

    1. Risk of Disease: Assessment of the patient's risk and personalized insights.
    2. Reassuring & Compassion: Comforting and empathetic message to reassure the patient.
    3. Eating Habits: Dietary recommendations tailored to the patient's condition.
    4. Exercise: Exercise recommendations based on the patient's diagnosis and health status.]

    Now, create a response following this structure based on the given inputs.
    """

    # Combine all parts of the template
    full_template = base_template + optional_template + output_template

    # Return the formatted template
    return full_template.format(
        age=age,
        sex=sex,
        diagnosis=diagnosis,
        cause=cause,
        symptom=symptom,
        treatment=treatment
        )


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

Optional input: these are optional inputs of patient. Please consider these optional information if needed. if 'None', you must not consider it.
    - history (bool or None){history}: Previous diagnosis status.
    - family_history (bool or None){family_history}: Family history of stomach cancer.
    - past_procedure (bool or None){past_procedure}: History of upper gastrointestinal procedures/surgeries.
    - past_infection (bool or None){past_infection}: History of H. pylori infection.
    - obesity (bool or None){obesity}: Whether the patient is obese.
    - coffee (bool or None){coffee}: Consumption of coffee.
    - alcohol (bool or None){alcohol}: Consumption of alcohol.
    - smoking (bool or None){smoking}: Consumption tobacco.
    - overeating (bool or None){overeating}: Whether the patient practices binge eating or overeating.
    - spicy_salting_fatty_food (bool or None){spicy_salting_fatty_food}: Consumption of spicy, salty, or fatty foods.
    - exercise (bool or None){exercise}: Whether the patient exercises or not.

Output:
1. Risk of disease: The risk of disease should present the risk for the disease diagnosed by the patient. In particular, the gender and age of the patient should be considered. For example, the risk of disease in general will increase due to decreased gastrointestinal function and accompanying chronic diseases as they are older. In addition, more specifically, for example, in the case of acute gastritis, the risk of acute gastritis will be high in young people due to frequent drinking, irregular eating habits, and stress. In addition, men will have a high risk of acute gastritis due to factors such as smoking, drinking, and stress, and women may have a high risk of acute gastritis due to factors such as hormonal changes, pregnancy, and childbirth. In this way, the risk of considering age or gender should be presented for each disease.
2. Reassuring & compassionation: Give compassionate and reassuring comment for patient diagnosed condition with "Diagnosis".
e.g.) I told you that gastritis can develop into stomach cancer, but the possibility is significantly less, so if you pay attention to your lifestyle from now on, it can be prevented enough. ^^
3. Eating habits: It presents detailed eating habits that should be good for the patient to have according to the disease diagnosed by the patient, or that the patient should abandon.
4. Exercise: The exercise habits that would be good for the patient to have according to the disease diagnosed by the patient, and the exercise that is helpful are solved along with the exercise method.


Now, create a response following this structure based on the given inputs.
"""

# Define the function to generate an answer
def generate_answer(age, sex, diagnosis, cause, symptom, treatment, optional_inputs):
    prompt = template.format(
        age=age, 
        sex=sex, 
        diagnosis=diagnosis, 
        cause=cause, 
        symptom=symptom, 
        treatment=treatment,
        history=optional_inputs["history"],
        family_history=optional_inputs["family_history"],
        past_procedure=optional_inputs["past_procedure"],
        past_infection=optional_inputs["past_infection"],
        obesity=optional_inputs["obesity"],
        coffee=optional_inputs["coffee"],
        alcohol=optional_inputs["alcohol"],
        smoking=optional_inputs["smoking"],
        overeating=optional_inputs["overeating"],
        spicy_salting_fatty_food=optional_inputs["spicy_salting_fatty_food"],
        exercise=optional_inputs["exercise"])
    response = pipe(prompt)
    
    # Extract the answer part from the generated text
    generated_text = response[0]['generated_text']
    
    # Remove the input prompt from the generated output
    answer = generated_text[len(prompt):].strip() 
    return answer

################################################# Input Section ###########################################
# modify here for another manual inference example and further utilization.

age = 45
sex = "Male"
decision = "크기가 큰 위점막하종양"
comment = "위내시경검사에서 위에 점막하종양이 있습니다. 점막하종양이란 위점막 아래에서 발생한 종양을 말하며, 이 때문에 위벽 일부가 부풀어 올라온 것처럼 보입니다. 점막하 종양은 종류에 따라 아무런 치료가 필요하지 않는 경우부터 수술적 절제가 필요한 경우까지 치료방법이 매우 다양합니다. 정확한 진단 및 적절한 치료 방침 결정을 위해 의료기관을 방문하시어 진료 상담을 받으시길 바랍니다."
diagnosis, cause, symptom, treatment = return_diagnosis(decision, comment)
optional_inputs = get_optional_inputs(history=True, family_history=None, past_procedure=None, past_infection=None, obesity=None,
                        coffee=None, alcohol=None, smoking=True, overeating=True, spicy_salting_fatty_food=None, exercise=None)

template = make_template(age, sex, diagnosis, cause, symptom, treatment, optional_inputs)
print(template)

############################################### Generation function ########################################

# Generate answer for the single input
answer = generate_answer(
    age, sex, diagnosis, cause, symptom, treatment, optional_inputs)


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

