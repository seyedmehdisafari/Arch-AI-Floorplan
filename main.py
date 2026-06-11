from src.train import train_model

DATA_DIR = "./data/Img"

def main():
    # Parameters configured for an initial benchmark test
    EPOCHS = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    
    print("--- Starting Text2Plan-AI Training Pipeline ---")
    train_model(
        data_dir=DATA_DIR, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        lr=LEARNING_RATE
    )

if __name__ == "__main__":
    main()