from src.process_data import get_and_send_prompts_in_batches


def main():
    get_and_send_prompts_in_batches(
        "mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", "train", 10, "results"
    )


if __name__ == "__main__":
    main()
