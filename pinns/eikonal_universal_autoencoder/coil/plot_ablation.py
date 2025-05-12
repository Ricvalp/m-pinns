from pinns.eikonal_autodecoder.plot import plot_ablation, plot_metrics_separately
import pandas as pd

if __name__ == "__main__":
    # Read both CSVs
    mpinn_df = pd.read_csv("pinns/eikonal_autodecoder/coil/ablation.csv")
    deltapinn_df = pd.read_csv("pinns/eikonal_autodecoder/coil/deltapinn_ablation.csv")

    # Combine the data
    combined_df = pd.merge(
        mpinn_df[["N", "mpinn_corr", "mpinn_mse", "seed"]],
        deltapinn_df[["N", "deltapinn_corr", "deltapinn_mse", "seed"]],
        on=["N", "seed"],
        how="outer",
    )

    # Save combined data
    combined_df.to_csv(
        "pinns/eikonal_autodecoder/coil/combined_ablation.csv", index=False
    )

    # Create plots
    plot_ablation(
        "pinns/eikonal_autodecoder/coil/ablation.csv",
        "pinns/eikonal_autodecoder/coil/deltapinn_ablation.csv",
        name="ablation_comparison",
    )

    plot_metrics_separately(
        "pinns/eikonal_autodecoder/coil/ablation.csv",
        "pinns/eikonal_autodecoder/coil/deltapinn_ablation.csv",
        name="ablation_comparison",
    )
