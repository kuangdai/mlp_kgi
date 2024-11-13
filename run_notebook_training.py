import os
import subprocess
import sys


def modify_script(file_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"File '{file_path}' not found.")
        return

    # Read the file contents and modify lines as needed
    with open(file_path, 'r') as file:
        lines = file.readlines()

    modified = False
    for i, line in enumerate(lines):
        # Modify specified lines if they exist
        if "plt.rcParams['text.usetex'] = True" in line:
            lines[i] = line.replace("plt.rcParams['text.usetex'] = True", "plt.rcParams['text.usetex'] = False")
            modified = True
        elif "from tqdm.notebook import trange" in line:
            lines[i] = line.replace("from tqdm.notebook import trange", "from tqdm import trange")
            modified = True
        elif "reproduce_paper = False" in line:
            lines[i] = line.replace("reproduce_paper = False", "reproduce_paper = True")
            modified = True
        elif line.strip() in ["# ## Analysis", "# # Analysis"]:
            lines[i] = 'assert False, "Train Finish. Skip Analysis."\n'
            modified = True

    # Write the modified contents back to the file if any modifications were made
    if modified:
        with open(file_path, 'w') as file:
            file.writelines(lines)
        print(f"Modifications applied to '{file_path}'.")
    else:
        print(f"No specified lines found in '{file_path}', no modifications made.")


def run_notebook(notebook_path):
    # Convert notebook to Python script
    if not os.path.isfile(notebook_path):
        print(f"Notebook file '{notebook_path}' not found.")
        return

    try:
        output_file = notebook_path.replace(".ipynb", ".py")
        convert_command = f"jupyter nbconvert --to script {notebook_path}"
        subprocess.run(convert_command, shell=True, check=True)
        print(f"Converted '{notebook_path}' to '{output_file}'.")

        # Modify the script with specified changes
        modify_script(output_file)

        # Run the modified Python script
        run_command = f"python {output_file}"
        subprocess.run(run_command, shell=True, check=True)
        print(f"Executed '{output_file}' successfully.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print("Please ensure nbconvert is installed (`pip install nbconvert`).")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_notebook_training.py <notebook_path>")
    else:
        run_notebook(sys.argv[1])
