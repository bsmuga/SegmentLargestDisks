# Segment Largest Disks

## Development Plan (TODO)

*   **Phase 1: Data Generation**
    *   [ ] Analyze the existing `src/dataset.py` to identify areas for improvement.
    *   [ ] Implement a robust synthetic dataset generation class with clear and well-documented logic.
    *   [ ] Write comprehensive unit tests for the dataset generation to ensure correctness.

*   **Phase 2: Segmentation Metrics**
    *   [ ] Review and refactor the existing `src/confusion_matrix.py`.
    *   [ ] Implement a pixel-wise confusion matrix suitable for image segmentation tasks.
    *   [ ] Add unit tests for the segmentation metrics.

*   **Phase 3: U-Net in JAX**
    *   [ ] Learn the fundamentals of JAX by implementing a simple neural network.
    *   [ ] Re-implement the U-Net architecture in JAX.
    

*   **Phase 4: Main Script**
    *   [ ] Refactor the `src/main.py` script to integrate the new dataset and metrics.
    *   [ ] Ensure the main script is clean, well-documented, and easy to run.
    *   [ ] Train and evaluate the JAX U-Net on the synthetic dataset.