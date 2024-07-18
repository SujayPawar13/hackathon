
import speech_recognition as sr
import pyttsx3
import streamlit as st
import torch
import string
from transformers import BertTokenizer, BertForMaskedLM, BertForNextSentencePrediction,  GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from simpletransformers.language_modeling import LanguageModelingModel,LanguageModelingArgs
from simpletransformers.language_generation import LanguageGenerationModel, LanguageGenerationArgs


### st.cache_data or st.cache_resource
##@st.cache()
@st.cache_data()
def load_model(model_name):
  try:
    if model_name.lower() == "bert":
      bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
      bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
      return bert_tokenizer,bert_model
  except Exception as e:
    pass


#use joblib to fast your function

def decode(tokenizer, pred_idx, top_clean):
  ignore_tokens = string.punctuation + '[PAD]'
  tokens = []
  for w in pred_idx:
    token = ''.join(tokenizer.decode(w).split())
    if token not in ignore_tokens:
      tokens.append(token.replace('##', ''))
  return '\n'.join(tokens[:top_clean])


def encode(tokenizer, text_sentence, add_special_tokens=True):
  text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
  if tokenizer.mask_token == text_sentence.split()[-1]:
    text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
  return input_ids, mask_idx



def get_all_predictions(text_sentence, top_clean=5):
    # ========================= BERT =================================
  input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
  with torch.no_grad():
    predict = bert_model(input_ids)[0]
  bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
  return {'bert': bert}


def get_prediction_eos(input_text):
  try:
    input_text += ' <mask>'
    res = get_all_predictions(input_text, top_clean=int(top_k))
    return res
  except Exception as error:
    pass


def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

def convertSpeechToText():
  MyText = ""

  # while (1) or MyText =="":
  #while MyText == "":

    # Exception handling to handle
    # exceptions at the runtime
  try:
    print("========= Inside while=========" + MyText)
    # use the microphone as source for input.
    with sr.Microphone() as source2:

      # wait for a second to let the recognizer
      # adjust the energy threshold based on
      # the surrounding noise level
      r.adjust_for_ambient_noise(source2, duration=0.2)

      print("========= listens for the user's input ")
      # listens for the user's input
      audio2 = r.listen(source2)

      # Using google to recognize audio
      MyText = r.recognize_google(audio2)
      #MyText = MyText.lower()
      speech_txt = MyText.lower()
      #speech_txt = speech_txt

      print("Did you say ", speech_txt)
      SpeakText(speech_txt)

      return speech_txt

  except sr.RequestError as e:
    print("Could not request results; {0}".format(e))

  except sr.UnknownValueError:
    print(sr)
    print("unknown error occurred")



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

  st.title("DB Hackathon Dementia - Team: Tech Titans")
  #st.logo("Tech Titans")
  speech_txt="DB Hackathon Dementia - Team: Tech Titans"
  r = sr.Recognizer()

  #speech_txt = convertSpeechToText()
  print("speech coverted text="+ speech_txt)
  st.sidebar.text("Next Sentence Prediction:")
  model_name = st.sidebar.selectbox(label='Select Model to Apply for sentence',  options=['GPT', 'XLNET'], index=0,  key = "model_name")

  st.sidebar.markdown("------------------------------------")
  st.sidebar.text("Next Word Prediction:")
  top_k = st.sidebar.slider("How many words do you need", 1, 25, 1)  # some times it is possible to have less words
  print(top_k)
  model_name_word = st.sidebar.selectbox(label='Select Model to Apply for word', options=['BERT', 'XLNET'], index=0,
                                         key="model_name_word")

  st.button("Record", on_click=convertSpeechToText, args=[])
  #input_text = st.text_area("Enter your text here:")
  input_text = st.text_area("Enter your text here:", value=speech_txt)


  #click outside box of input text to get result
  res = model.generate(input_text)
  answer_as_string = res

  ################# Word ###############
  bert_tokenizer, bert_model = load_model(model_name_word)
  # input_text_word = st.text_area("Enter your text here")
  # click outside box of input text to get result
  res_word = get_prediction_eos(input_text)
  answer = []
  print(res_word['bert'].split("\n"))
  for i in res_word['bert'].split("\n"):
    answer.append(i)
  answer_as_string_word = "    ".join(answer)
  ################# Word ###############

  st.text_area("Predicted Sentence is Here:",answer_as_string,key="predicted_list")

  st.text_area("Predicted Word List is Here:", answer_as_string_word, key="predicted_list_word")

except Exception as e:
  print("SOME PROBLEM OCCURED!!!")
