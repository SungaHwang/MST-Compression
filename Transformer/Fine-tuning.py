import os
import logging
import datetime
import argparse
from transformers import BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

def setup_logging(model_name):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_directory = "Transformer/log"
    log_filename = f"fine-tuning-{model_name}-{current_time}.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=os.path.join(log_directory, log_filename))
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging started for model: {model_name}")

    return logger

def preprocess_function(examples, model_name):
    tokenizer = BertTokenizer.from_pretrained(model_name) if "bert" in model_name else RobertaTokenizer.from_pretrained(model_name)
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True)

def main(args):
    logger = setup_logging(args.model_name)

    # 모델 선택
    if "bert" in args.model_name:
        model = BertForSequenceClassification.from_pretrained(args.model_name)
    elif "roberta" in args.model_name:
        model = RobertaForSequenceClassification.from_pretrained(args.model_name)
    else:
        raise ValueError("Unsupported model name:", args.model_name)

    # GLUE 데이터셋 불러오기
    glue_dataset = load_dataset("glue", "mrpc")

    train_dataset = glue_dataset["train"].map(lambda x: preprocess_function(x, args.model_name), batched=True)
    eval_dataset = glue_dataset["validation"].map(lambda x: preprocess_function(x, args.model_name), batched=True)

    # fine-tuning을 위한 학습 인자 설정
    training_args = TrainingArguments(
        per_device_train_batch_size=8,          # 각 장치당 훈련 배치 크기
        per_device_eval_batch_size=8,           # 각 장치당 검증 배치 크기
        num_train_epochs=5,                     # 훈련 에폭 수
        evaluation_strategy="steps",            # 검증을 언제 수행할 것인지 설정
        eval_steps=500,                         # 매 500 스텝마다 검증 수행
        logging_dir="./logs",                   # 로그 디렉토리
        output_dir=f"./{args.model_name}_model",# 모델 출력 디렉토리
        save_strategy="epoch",                  # 언제 모델을 저장할 것인지 설정
        save_total_limit=2,                     # 저장할 체크포인트의 최대 개수
        gradient_accumulation_steps=2,          # 그래디언트 누적 스텝 수
        learning_rate=2e-5,                     # 학습률
        weight_decay=0.01,                      # 가중치 감쇠
        warmup_steps=500,                       # 워머핑 스텝
    )

    # Trainer 객체 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # fine-tuning 및 평가 실행
    logger.info("Training started")
    trainer.train()
    logger.info("Training finished")

    logger.info("Evaluation started")
    evaluation_results = trainer.evaluate()
    logger.info("Evaluation finished")

    # 평가 결과 출력 및 로그 저장
    logger.info("Evaluation results:")
    for key, value in evaluation_results.items():
        logger.info(f"{key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning BERT or RoBERTa on GLUE dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the BERT or RoBERTa model to use")
    args = parser.parse_args()
    main(args)
