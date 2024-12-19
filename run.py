import sys

print("sys.path = ", sys.path)
import sglang as sgl
import os


def print_environment_info():
    # CUDA-related environment variables
    cuda_vars = ["CUDA_VISIBLE_DEVICES", "CUDA_HOME", "LD_LIBRARY_PATH", "CUDNN_PATH"]

    print("\n=== CUDA Environment Variables ===")
    for var in cuda_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

    # Add CUDA version check
    try:
        import torch
        import platform

        print(f"\nPython Version: {platform.python_version()}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
    except ImportError:
        print("\nTorch not installed - cannot check CUDA version")
    except Exception as e:
        print(f"\nError checking CUDA version: {str(e)}")

    # C++-related environment variables
    cpp_vars = ["CXX", "CXXFLAGS", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH"]

    print("\n=== C++ Environment Variables ===")
    for var in cpp_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

    # Python-related environment variables
    python_vars = ["PYTHONPATH", "PYTHONHOME", "PYTHON_VERSION", "VIRTUAL_ENV"]

    print("\n=== Python Environment Variables ===")
    for var in python_vars:
        print(f"{var}: {os.environ.get(var, 'Not set')}")

    # Print sys.path
    print("\n=== Python sys.path ===")
    for path in sys.path:
        print(path)


def main():
    # Print environment information
    print_environment_info()

    print("\n=== Starting LLM Generation ===")
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
