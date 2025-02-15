# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union

import pickle , csv
import numpy as np
import torch
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase
from transformers.trainer_utils import is_main_process


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If passed, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to the maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    dic_sources = 'data/'
    dic_save    = 'save/'

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    print("File: {}".format( data_args.train_file ) )
    print("File: {}".format( data_args.validation_file) )
    if data_args.train_file is not None or data_args.validation_file is not None:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)
    else:
        datasets = load_dataset("swag", "regular")

    # Load pretrained model and tokenizer

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    torch.cuda.empty_cache()
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # pikle functions
    def pickleOpen( filename ):
        file_to_read = open( filename , "rb" )
        p = pickle.load( file_to_read )
        return p

    def pickleStore( savethings , filename ):
        dbfile = open( filename , 'wb' )
        pickle.dump( savethings , dbfile )
        dbfile.close()
        return

    # Preprocessing the datasets.
    def preprocess_function( data ):

        docs_dict = pickleOpen( dic_save + "docs_dict.pkl" )
        first_sentences = [ [ query_content ] * 4 for query_content in data['query_content'] ]
        second_sentences = []

        for tup in zip( data[ 'query_content' ] , data[ 'answer' ] ):
            pn_list = tup[1].split()
            for doc_name in pn_list:
                second_sentences.append( [ f"{tup[0]} {docs_dict[ doc_name ]}" ] )

        # Flatten out
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=data_args.max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        # Un-flatten
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


    def test_preprocess_function( data ):

        docs_dict = pickleOpen( dic_save + "docs_dict.pkl" )
        first_sentences  = [ [ query_content ] * 4 for query_content in data[ 'query_content' ] ]
        second_sentences = []

        for tup in zip( data[ 'query_content' ] , data[ 'top1000' ] ):
            pn_list = tup[1].split()
            for doc_name in pn_list:
                a = docs_dict[ doc_name ] if doc_name in docs_dict.keys() and docs_dict[ doc_name ] is not None else " None "
                second_sentences.append( [ f"{tup[0]} {a}" ] )

        # Flatten out
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            truncation=True, 
            max_length=data_args.max_seq_length,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Un-flatten
        return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}


    # Data collator
    data_collator = (
        default_data_collator if data_args.pad_to_max_length else DataCollatorForMultipleChoice(tokenizer=tokenizer)
    )

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    # Data tokenized
    logger.info("*** Preprocessing Train data ***")
    tokenized_datasets = datasets.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Preprocessing the test datasets.
    logger.info("*** Preprocessing Test data ***")
    try:
        test_data_files = {}
        test_data_files["train"] = dic_save + "test.csv"
        testdata = load_dataset("csv", data_files=test_data_files)
        # testdata = load_dataset( "csv" , data_files=dic_save + "test.csv" )
        tokenized_test_datasets = testdata.map(
            test_preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    except:
        logger.info("*** An exception occurred: test data error ***")

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    torch.cuda.empty_cache()
    # Training
    if training_args.do_train:
        train_result = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

        try:
            print( train_result )
            output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
            if trainer.is_world_process_zero():
                with open(output_train_file, "w") as writer:
                    logger.info("***** Train results *****")
                    for key, value in sorted(train_result.metrics.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")
        except:
            logger.info("*** An exception occurred: Save train record error ***")

        # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
        if trainer.is_world_process_zero():
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Save Trainer Foreahead
    try:
        pickleStore( train_result , dic_save + "train_result.pkl" )
    except:
        logger.info("*** An exception occurred: Save train model error ***")

    # Evaluation
    results = {}
    try:
        if training_args.do_eval:
            logger.info("*** Evaluate ***")

            results = trainer.evaluate()

            output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
            if trainer.is_world_process_zero():
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key, value in sorted(results.items()):
                        logger.info(f"  {key} = {value}")
                        writer.write(f"{key} = {value}\n")    
    except:
        logger.info("*** An exception occurred: Evaluation error ***")

    # Test
    try:
        logger.info("*** Predict Test dataset ***")
        trainer_test_result = trainer.predict( test_dataset=tokenized_test_datasets["train"] )
        print( "trainer_test_result type: {}".format( type( trainer_test_result ) ) )
    except:
        logger.info("*** An exception occurred: Predict Test error ***")

    try:
        output_test_file = os.path.join( training_args.output_dir , "test_results.txt" )
        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in sorted(trainer_test_result[2].items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n") 
    except:
        logger.info("*** An exception occurred: Test Result print error ***")

    try:
        logger.info("*** Save Predict Test Result ***")
        pickleStore( trainer_test_result , dic_save + "trainer_test_result.pkl" )
    except:
        logger.info("*** An exception occurred: Save Predict error ***")

    # predict for training alpha
    try:
        logger.info("*** Predict for Training Alpha ***")
        alphadata = load_dataset( "csv" , data_files=dic_save + "train_for_alpha_train.csv" )
        tokenized_alpha_datasets = alphadata.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    except:
        logger.info("*** An exception occurred: Predict Alpha Data error first time, try again ***")
        try:
            alpha_data_files = {}
            alpha_data_files["train"] = dic_save + "train_for_alpha_train.csv"
            alphadata = load_dataset("csv", data_files=test_data_files)
            tokenized_alpha_datasets = alphadata.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )
        except:
            logger.info("*** An exception occurred: Predict Alpha Data error 2 times ***")

    try:
        trainer_alpha_result = trainer.predict( test_dataset=tokenized_alpha_datasets["train"] )
    except:
        logger.info("*** An exception occurred: Predict Alpha error ***")

    try:
        logger.info("*** Save Predict Alpha Result ***")
        pickleStore( trainer_alpha_result , dic_save + "trainer_alpha_result.pkl" )
    except:
        logger.info("*** An exception occurred: Save Predict alpha error ***")

    # calculate documents ranking for test
    logger.info("*** Caculate Reranking Result ***")

    def softmax( vector ):
        e = np.exp( vector )
        return e / e.sum()

    try:
        save = {}
        tmp  = {}
        new  = {}
        pre_result = trainer_test_result[0]
        with open( dic_save + 'test.csv' , newline='' ) as csvfile:
            spamreader = csv.reader( csvfile , delimiter=',' )
            next(spamreader)
            for c , row in enumerate( spamreader ):
                if row[0] not in save.keys():
                    save[ row[0] ] = {}
                for tup in zip( row[2].split() , pre_result[c].tolist() ):
                    save[ row[0] ][ tup[0] ] = tup[1]
            # sort
            save = { query_name : dict( sorted( d_v.items(), key=lambda item: item[1] , reverse=True ) ) for query_name , d_v in save.items() }
            # compute softmax
            for query_name , d_v in save.items():
                tmp[ query_name ] = softmax( np.array( list( save[ query_name ].values() ) , dtype=np.float64 ) ).tolist()
            save = { query_name : dict( zip( list( save[ query_name ].keys() ) , tmp[ query_name ] ) ) for query_name in save.keys() }
        
        que_top_dict = pickleOpen( dic_save + "test_que_top_dict.pkl" )
        for query_name , d_v in que_top_dict.items():
            for doc_name , softmax_score in d_v.items():
                new[ query_name ][ doc_name ] = np.log2( save[ query_name ][ doc_name ] ) + 1.15 * np.log2( softmax_score )
        pickleStore( new , dic_save + "test_ques_docs_reranking.pkl" )

        with open( dic_save + "test_ques_docs_reranking.csv" , "a" ) as writefile:
            writefile.write( "query_id,ranked_doc_ids\n" )
            for query_name , d_v in new.items():
                append = ",".join( [ str( query_name ) , " ".join( sorted( d_v , key=d_v.get , reverse=True ) ) ] )
                writefile.write( append + "\n" )
    except:
        logger.info("*** Caculate Reranking Result Error ***")

    # DONE
    logger.info("*** Done! ***")

    return


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
