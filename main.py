from src.get_logprobs import save_logprobs
from src.process_data import get_and_send_prompts_in_batches


def main():
    get_and_send_prompts_in_batches(
        "mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", "train", 10, "results.csv"
    )
    save_logprobs("results.csv", "logprobs_results.csv", 10)


if __name__ == "__main__":
    main()
