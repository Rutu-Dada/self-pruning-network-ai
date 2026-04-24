import matplotlib.pyplot as plt

def compute_sparsity(gates, threshold=1e-2):
    total = gates.numel()
    zero = (gates < threshold).sum().item()
    return 100 * zero / total


def plot_gates(gates, name="plot"):
    gates = gates.detach().cpu().numpy()

    plt.hist(gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Values")
    plt.ylabel("Frequency")

    plt.savefig(f"plots/{name}.png")
    plt.close()