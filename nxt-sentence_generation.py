# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 14:38:06 2023

@author: Arnav
"""

#import panel
import streamlit as st
import torch
import string

from simpletransformers.language_modeling import LanguageModelingModel,LanguageModelingArgs
from simpletransformers.language_generation import LanguageGenerationModel, LanguageGenerationArgs

# Editing Configurations
model_args = LanguageModelingArgs()
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.num_train_epochs = 2
model_args.best_model_dir = "outputs/best_model"
model_args.save_best_model =True
model_args.train_batch_size = 8
model_args.dataset_type = "simple"
model_args.mlm = False  # mlm must be False for CLM
model_args.vocab_size = 50257

#Train and test file loading
train_file = "train.txt"
test_file = "test.txt"

#Language Model Loading it can either GPT2, BERT, ELECTRA etc.
# model = LanguageModelingModel(
#     "gpt2", "gpt2", args=model_args, train_files=train_file
# )


model = LanguageModelingModel(
    "gpt2", "gpt2", args=model_args, use_cuda=False
#"gpt2", "gpt2", args=model_args, use_cuda=False
)

# Train the model
#model.train_model(train_file, eval_file=test_file)

# Evaluate the model
#result= model.eval_model(test_file)

print("===== model result =====")
#print(str(result))

# Model settings

Language_gen_args = LanguageGenerationArgs()
Language_gen_args.max_length = 100

# Language Generation
#model = LanguageGenerationModel("gpt2", "/content/outputs/checkpoint-8358-epoch-2", use_cuda=False)
model = LanguageGenerationModel("gpt2", "gpt2", use_cuda=False)
# output = model.generate(
#   "Despite the recent successes of deep learning, such models are still far from some human abilities like learning from few examples, reasoning and explaining decisions. In this paper, we focus on organ annotation in medical images and we introduce a reasoning framework that is based on learning fuzzy relations on a small dataset for generating explanations. Given a catalogue of relations, it efficiently induces the most relevant relations and combines them for building constraints in order to both solve the organ annotation task and generate explanations. We test our approach on a publicly available dataset of medical images where several organs are already segmented. A demonstration of our model is proposed with an example of explained annotations. It was trained on a small training set containing as few as a couple of examples. ")
#
# print("===== model output =====")
# print(output)


try:

  st.title("Next Sentence Prediction with Pytroch")
  #st.logo("Tech Titans")

  st.sidebar.text("Next Sentence Prediction")
  # top_k = st.sidebar.slider("How many words do you need", 1 , 25, 1) #some times it is possible to have less words
  # print(top_k)
  model_name = st.sidebar.selectbox(label='Select Model to Apply',  options=['GPT', 'BERT'], index=0,  key = "model_name")

  input_text = st.text_area("Enter your text here")
  #click outside box of input text to get result
  res = model.generate(input_text)

  answer_as_string = res
  st.text_area("Predicted List is Here",answer_as_string,key="predicted_list")

except Exception as e:
  print("SOME PROBLEM OCCURED")

