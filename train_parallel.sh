#!/bin/bash

# Check if all required arguments were provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <number_of_instances> <version>"
    echo "Example: $0 20 v26"
    exit 1
fi

# Arguments
N=$1
VERSION=$2

# Validate number of instances is a positive number
if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
    echo "Error: First argument must be a positive number"
    exit 1
fi

# Validate version starts with 'v'
if ! [[ "$VERSION" =~ ^v[0-9]+$ ]]; then
    echo "Error: Version must be in format 'v<number>' (e.g., v26)"
    exit 1
fi

# Get current datetime for log files
DATETIME=$(date "+%Y%m%d_%H%M")

# Counter for naming log files
counter=1

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting $N training instances with version $VERSION..."

# Start N instances in parallel
for i in $(seq 1 $N); do
    LOG_FILE="logs/train_${VERSION}_${DATETIME}_${counter}.log"
    python train.py -ev $VERSION 2>&1 > "$LOG_FILE" &
    echo "Started instance $counter with version $VERSION (log: $LOG_FILE)"
    ((counter++))
done

# Wait for all background processes to complete
wait

echo "All training instances have completed"

# Check exit status of all processes
for job in $(jobs -p); do
    wait $job || echo "Process $job failed"
done

echo "Log files are stored in the logs directory with format: train_<version>_<datetime>_<instance>.log"
