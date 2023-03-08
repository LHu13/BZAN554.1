# Info on RAM Usage
import psutil

# Retrieves system memory usage stats
mem_stats = psutil.virtual_memory()

# Print memory stats
print(f"Total Memory: {mem_stats.total} bytes") #Total Physical Memory
print(f"Used Memory: {mem_stats.used} bytes") #Memory currently in use
print(f"Free Memory: {mem_stats.available} bytes") #Memory available for use


#Info on Training Time
import time

start_time = time.time() #Current time in seconds since Epoch

# Training the machine learning model
end_time = time.time() #End Time

training_time = end_time - start_time #Total Train Time

print(f"Training time: {training_time:.2f} seconds") #Prints train time to 2 decimals
