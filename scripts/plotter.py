import matplotlib.pyplot as plt

def plot_metrics(history, facs_metric, emotion_metric):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Plot training and validation loss for FACS codes
    axes[0, 0].plot(history.history['facs_output_loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_facs_output_loss'], label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('FACS Output')
    axes[0, 0].grid(alpha=0.7, linestyle='--')
    axes[0, 0].legend()


    # Plot training and validation accuracy for FACS codes
    axes[0, 1].plot(history.history[facs_metric], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_' + facs_metric], label='Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('FACS Output')
    axes[0, 1].grid(alpha=0.7, linestyle='--')
    axes[0, 1].legend()

    # Plot training and validation loss for emotion labels
    axes[1, 0].plot(history.history['emotion_output_loss'], label='Training Loss')
    axes[1, 0].plot(history.history['val_emotion_output_loss'], label='Validation Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Emotion Output')
    axes[1, 0].grid(alpha=0.7, linestyle='--')
    axes[1, 0].legend()

    # Plot training and validation accuracy for emotion labels
    axes[1, 1].plot(history.history['emotion_output_f1M'], label='Training Accuracy')
    axes[1, 1].plot(history.history['val_emotion_output_f1M'], label='Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Emotion Output')
    axes[1, 1].grid(alpha=0.7, linestyle='--')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()