import sglang as sgl
import os


def main():
    # Initialize the engine here, inside __main__ and after freeze_support
    llm = sgl.Engine(
        model_path=os.path.expandvars("$HOME/.llms/models/Qwen2-7B-Instruct")
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "Solve the following equation: 2x + 3 = 7",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    outputs = llm.generate(prompts, sampling_params)

    # Open a file to write the results
    with open("generation_results.txt", "w", encoding="utf-8") as f:
        for prompt, output in zip(prompts, outputs):
            # Print to console
            print("===============================")
            print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

            # Write to file
            f.write("===============================\n")
            f.write(f"Prompt: {prompt}\nGenerated text: {output['text']}\n")

    llm.shutdown()


if __name__ == "__main__":
    main()
