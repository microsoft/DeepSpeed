set -x

model_dir="./transformers/src/transformers/models/"
models=("electra" "roberta" "t5" "wav2vec2" "bert")
modules=("ElectraLayer" "RobertaLayer" "T5Block" "Wav2Vec2EncoderLayer" "BertLayer")

t5_modules=("T5LayerNorm" "T5DenseActDense" "T5DenseGatedActDense" "T5LayerFF" "T5Attention" "T5LayerSelfAttention" "T5LayerCrossAttention" "T5Block" "T5PreTrainedModel" "T5Stack" "T5Model" "T5ForConditionalGeneration" "T5EncoderModel")

bert_modules=("BertEmbeddings" "BertSelfAttention" "BertSelfOutput" "BertAttention" "BertIntermediate" "BertOutput" "BertLayer" "BertEncoder" "BertPooler" "BertPredictionHeadTransform" "BertLMPredictionHead" "BertOnlyMLMHead" "BertOnlyNSPHead" "BertPreTrainingHeads" "BertPreTrainedModel" "BertForPreTrainingOutput" "BertModel" "BertForPreTraining" "BertLMHeadModel" "BertForMaskedLM" "BertForNextSentencePrediction" "BertForSequenceClassification" "BertForMultipleChoice" "BertForTokenClassification" "BertForQuestionAnswering")

electra_modules=("ElectraEmbeddings" "ElectraSelfAttention" "ElectraSelfOutput" "ElectraAttention" "ElectraIntermediate" "ElectraOutput" "ElectraLayer" "ElectraEncoder" "ElectraDiscriminatorPredictions" "ElectraGeneratorPredictions" "ElectraPreTrainedModel" "ElectraForPreTrainingOutput" "ElectraModel" "ElectraClassificationHead" "ElectraForSequenceClassification" "ElectraForPreTraining" "ElectraForMaskedLM" "ElectraForTokenClassification" "ElectraForQuestionAnswering" "ElectraForMultipleChoice" "ElectraForCausalLM")

roberta_modules=("RobertaEmbeddings" "RobertaSelfAttention" "RobertaSelfOutput" "RobertaAttention" "RobertaIntermediate" "RobertaOutput" "RobertaLayer" "RobertaEncoder" "RobertaPooler" "RobertaPreTrainedModel" "RobertaModel" "RobertaForCausalLM" "RobertaForMaskedLM" "RobertaLMHead" "RobertaForSequenceClassification" "RobertaForMultipleChoice" "RobertaForTokenClassification" "RobertaClassificationHead" "RobertaForQuestionAnswering")

wav2vec2_modules=("Wav2Vec2ForPreTrainingOutput" "Wav2Vec2NoLayerNormConvLayer" "Wav2Vec2LayerNormConvLayer" "Wav2Vec2GroupNormConvLayer" "Wav2Vec2PositionalConvEmbedding" "Wav2Vec2SamePadLayer" "Wav2Vec2FeatureEncoder" "Wav2Vec2FeatureExtractor" "Wav2Vec2FeatureProjection" "Wav2Vec2Attention" "Wav2Vec2FeedForward" "Wav2Vec2EncoderLayer" "Wav2Vec2EncoderLayerStableLayerNorm" "Wav2Vec2Encoder" "Wav2Vec2EncoderStableLayerNorm" "Wav2Vec2GumbelVectorQuantizer" "Wav2Vec2Adapter" "Wav2Vec2AdapterLayer" "Wav2Vec2PreTrainedModel" "Wav2Vec2Model" "Wav2Vec2ForPreTraining" "Wav2Vec2ForMaskedLM" "Wav2Vec2ForCTC" "Wav2Vec2ForSequenceClassification" "Wav2Vec2ForAudioFrameClassification" "AMSoftmaxLoss" "TDNNLayer" "Wav2Vec2ForXVector")


#length=${#models[@]}
#for ((i=0;i<$length;i++)); do
#	python get_injection_policy.py -f ./transformers/src/transformers/models/${models[$i]}/modeling_${models[$i]}.py -m ${modules[$i]} 
#done

for module in ${t5_modules[@]}; do
        python get_injection_policy.py -f ./transformers/src/transformers/models/t5/modeling_t5.py -m $module
done

for module in ${bert_modules[@]}; do
        python get_injection_policy.py -f ./transformers/src/transformers/models/bert/modeling_bert.py -m $module
done

for module in ${electra_modules[@]}; do
        python get_injection_policy.py -f ./transformers/src/transformers/models/electra/modeling_electra.py -m $module
done

for module in ${roberta_modules[@]}; do
	python get_injection_policy.py -f ./transformers/src/transformers/models/roberta/modeling_roberta.py -m $module
done

for module in ${wav2vec2_modules[@]}; do
        python get_injection_policy.py -f ./transformers/src/transformers/models/wav2vec2/modeling_wav2vec2.py -m $module
done
