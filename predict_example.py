import torch
import sentencepiece as spm

from bit_prediction import WordSuggestionWrapper, LLM


def main():
    tokenizer = spm.SentencePieceProcessor()
    # tokenizer.load("models/tokenizer.model")

    ## Trained LLM instance
    llm = LLM.build("models/bit-llm", tokenizer=tokenizer)

    ## Wrap LLM instance to perform word (sub-word) predictions
    llm_wrap = WordSuggestionWrapper(
        llm, tokenizer, k=3, temperature=0.5, vary_temperature=1
    )

    texts = [
        "Hello! How is your day g",
        "This project allows me to ga",
        "After walking for 30 ",
    ]

    for text in reversed(texts):
        out, _, _ = llm_wrap.next_word(
            input_sentence=text, n_best=9
        )

        print(text)
        print(out)



if __name__ == "__main__":
    main()
