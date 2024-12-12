import matplotlib.pyplot as plt
import seaborn as sns

def plot_runtime_distribution(filename, output_file="runtime_distribution.png"):
    """
    Reads runtime data from a file, plots its distribution, and saves it as a PNG.

    :param filename: Name of the input file containing runtime values.
    :param output_file: Name of the output PNG file.
    """
    try:
        # Read runtime data from the file
        with open(filename, 'r') as file:
            runtimes = [float(line.strip()) for line in file if line.strip()]
        
        # Convert runtimes to microseconds for better visualization
        runtimes_microseconds = [runtime * 1e6 for runtime in runtimes]  # Convert seconds to microseconds
        
        # Plot the distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(runtimes_microseconds, kde=True, bins=50, color='blue', alpha=0.7)
        
        # Customize the graph
        plt.title('Runtime Distribution (Microseconds)', fontsize=14)
        plt.xlabel('Elapsed Time (microseconds)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the plot as a PNG file
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {output_file}")
        
        # Show the plot
        plt.show()
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except ValueError:
        print("Error: File contains non-numeric data.")

# Example usage
input_filename = "output.txt"  # Replace with your file name
output_filename = "runtime_distribution.png"  # Name of the output file
plot_runtime_distribution(input_filename, output_filename)

